from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder
from .instruction import EInsnAttrs, Instruction
from .instructions.load_tensor_infos import emit_load_tensor_infos
from .state import StackMachineState, InvalidStateException
from .structs import TensorInfo

from .instructions.arith import *
from .instructions.control import *
from .instructions.immediate import *
from .instructions.load_store import *
from .instructions.move import *
from .instructions.predication import *
from .instructions.reduction import *
from .instructions.special import *


MAX_STACK = 8
REG_WIDTH = 4
SMEM_PER_WARP = 8 * 4 * (2 + 5)


INSNS: dict[type[Instruction], int] = {
    Halt: 0,
    Nop: 1,
    FAdd: 2,
    FMul: 3,
    FMA: 4,
    FSub: 5,
    FRSub: 6,
    FNeg: 7,
    FDiv: 8,
    FRDiv: 9,
    FRem: 10,
    LoadImm: 11,
    FAddImm: 12,
    FMulImm: 13,
    FMAImm: 14,
    LoadGlobalF32: 15,
    StoreGlobalF32: 16,
    LoadGlobalF16: 17,
    StoreGlobalF16: 18,
    LoadGlobalBF16: 19,
    StoreGlobalBF16: 20,
    LoadGlobalU8: 21,
    Pop: 22,
    Pop2: 23,
    Dup: 24,
    DupX1: 25,
    DupX2: 26,
    Dup2: 27,
    FGt: 28,
    FLt: 29,
    FGe: 30,
    FLe: 31,
    FEq: 32,
    FNe: 33,
    FApprox: 34,
    Select: 35,
    WarpAllReduceSum: 36,
    WarpAllReduceMax: 37,
    WarpAllReduceMin: 38,
    WarpAllReduceProd: 39,
    FSqrt: 40,
    FSin: 41,
    FCos: 42,
    FTan: 43,
    FArcSin: 44,
    FArcCos: 45,
    FArcTan: 46,
    FArcTan2: 47,
    FPow: 48,
    FExp: 49,
    FExp2: 50,
    FLog: 51,
    FLog2: 52,
    FRSqrt: 53,
    FErf: 54,
}


def build_main_loop(LL: PlatformIRBuilder):
    # declare dynamic smem
    smem_base = ir.GlobalVariable(LL.module, ir.ArrayType(i8, 0), name='dynamic_smem', addrspace=3)
    smem_base.linkage = 'external'
    smem_base.align = 16
    
    # early exit warps beyond user scheduling
    with LL.if_then(LL.icmp_unsigned('>=', LL.logical_program_idx(), LL.arg(3)), False):
        LL.ret_void()
    
    smem_base = LL.gep(smem_base, [i32(0), LL.mul(i32(SMEM_PER_WARP), LL.thread_idx_y())])
    
    entry_bb = LL.block
    dispatch_bbs: dict[int, ir.Block] = {}
    dispatch_states: dict[int, StackMachineState] = {}
    for i in range(MAX_STACK + 1):
        dispatch_bbs[i] = LL.append_basic_block("dispatch.%d" % i)
        LL.position_at_end(dispatch_bbs[i])
        dispatch_states[i] = StackMachineState(LL, smem_base, MAX_STACK, REG_WIDTH, i)
    undef_bb = LL.append_basic_block("unreachable")
    
    def br_state(state: StackMachineState):
        sp = state.sp
        for s, t in zip(dispatch_states[sp].flat_regs(), state.flat_regs()):
            s.add_incoming(t, LL.block)
        return LL.branch(dispatch_bbs[sp])
    
    LL.position_at_end(entry_bb)
    entry_pc = LL.bitcast(LL.arg(0), i32.as_pointer())
    entry_opcode = LL.load(entry_pc)
    state = dispatch_states[0].clone()
    state.pc = entry_pc
    state.opcode = entry_opcode
    for i in range(MAX_STACK):
        state.stack[i] = [f32(ir.Undefined)] * REG_WIDTH
    emit_load_tensor_infos(LL, state)
    br_state(state)
    
    for i in range(MAX_STACK + 1):
        LL.position_at_end(dispatch_bbs[i])
        dispatch_switch = LL.switch(dispatch_states[i].opcode, undef_bb)
        dispatch_weights = [1]
        for Insn, opid in INSNS.items():
            insn_bb = LL.append_basic_block("%s.%d" % (Insn.__name__, i))
            LL.position_at_end(insn_bb)
            cstate = dispatch_states[i].clone()
            insn = Insn(LL, cstate)
            attrs = insn.attrs()
            try:
                insn.emit_self()
            except InvalidStateException:
                LL.unreachable()
            else:
                if not (attrs & EInsnAttrs.NoReturn):
                    br_state(cstate)
            dispatch_switch.add_case(opid, insn_bb)
            if attrs & EInsnAttrs.Unlikely:
                dispatch_weights.append(1)
            else:
                dispatch_weights.append(10)
        dispatch_switch.set_weights(dispatch_weights)
    
    LL.position_at_end(undef_bb)
    LL.unreachable()


GEvalFType = ir.FunctionType(void, [
    i32.as_pointer(),  # code
    TensorInfo.as_pointer(),  # tensor info
    i32,  # num of tensors
    i32,  # logical grid dim before dividing warps
])


def build_interpreter_main_nvptx() -> PlatformIRBuilder:
    LL = NVPTXIRBuilder.create_kernel_module(GEvalFType, "geval")
    build_main_loop(LL)
    return LL
