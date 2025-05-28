from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder
from .instruction import EInsnAttrs
from .instructions.load_store import *
from .instructions.control import *
from .instructions.arith import *
from .instructions.move import *
from .instructions.immediate import *
from .instructions.reduction import *
from .instructions.special import *
from .instructions.predication import *
from .state import InterpreterState, get_spec
from .structs import TensorInfo


insns: list[Instruction] = [
    Halt(),
    Nop(),  # LoadTensorInfos(),
    LoadGlobalF32(),
    StoreGlobalF32(),
    FAdd(),
    MovF1F0(),
    MovF2F0(),
    MovF3F0(),
    MovF0F1(),
    MovF2F1(),
    MovF3F1(),
    MovF0F2(),
    MovF1F2(),
    MovF3F2(),
    MovF0F3(),
    MovF1F3(),
    MovF2F3(),
    FMul(),
    FMA(),
    FSub(),
    FRSub(),
    FDiv(),
    FRDiv(),
    FNeg(),
    LoadF0Imm(),
    LoadF1Imm(),
    LoadF2Imm(),
    LoadF3Imm(),
    WarpAllReduceSum(),
    WarpAllReduceMax(),
    WarpAllReduceMin(),
    WarpAllReduceProd(),
    FRem(),
    FSqrt(),
    FSin(),
    FCos(),
    FTan(),
    FArcSin(),
    FArcCos(),
    FArcTan(),
    FArcTan2(),
    FPow(),
    FExp(),
    FExp2(),
    FLog(),
    FLog2(),
    FRSqrt(),
    FErf(),
    LoadGlobalF16(),
    StoreGlobalF16(),
    LoadGlobalBF16(),
    StoreGlobalBF16(),
    LoadGlobalU8(),
    FGe(),
    FGt(),
    FLe(),
    FLt(),
    FEq(),
    FNe(),
    FApprox(),
    Select(),
    FAddImm(),
    FMulImm(),
    FMAImm(),
]

ILP = 8
SMEM_PER_WARP = 8 * 4 * (2 + 5)


def build_main_loop(LL: PlatformIRBuilder):
    # Stages for main interpreter loop:
    # issue load next instruction from memory (but don't use) (probably in L1)
    # dispatch!
    # come back from instructions directly to dispatch. phi-in local vars
    ispec = get_spec(ILP)
    
    # declare dynamic smem
    smem_base = ir.GlobalVariable(LL.module, ir.ArrayType(i8, 0), name='dynamic_smem', addrspace=3)
    smem_base.linkage = 'external'
    smem_base.align = 16
    
    smem_base = LL.gep(smem_base, [i32(0), LL.mul(i32(SMEM_PER_WARP), LL.thread_idx_y())])
    
    entry_bb = LL.block
    dispatch_bb = LL.append_basic_block("dispatch")
    # back_bb = LL.append_basic_block("back")
    undef_bb = LL.append_basic_block("unreachable")
    
    LL.position_at_end(entry_bb)
    entry_pc = LL.bitcast(LL.arg(0), i32.as_pointer())
    entry_opcode = LL.load(entry_pc)
    
    # early exit warps beyond user scheduling
    with LL.if_then(LL.icmp_unsigned('>=', LL.logical_program_idx(), LL.arg(3)), False):
        LL.ret_void()

    state = InterpreterState([init for name, init in ispec.flat_reg_inits()], i32(0), ispec, smem_base)
    LoadTensorInfos().emit(LL, state, ispec)
    post_entry_bb = LL.block
    
    LL.branch(dispatch_bb)
    
    LL.position_at_end(dispatch_bb)  # in: post_entry, insns
    regs = [LL.phi(init.type, name) for name, init in ispec.flat_reg_inits()]
    for reg, assn_reg in zip(regs, state.assn_regs):
        reg.add_incoming(assn_reg, post_entry_bb)
    
    pc = LL.phi(i32.as_pointer())
    pc.add_incoming(entry_pc, post_entry_bb)
    opcode = LL.phi(i32)
    opcode.add_incoming(entry_opcode, post_entry_bb)
    
    dispatch_switch = LL.switch(opcode, undef_bb)
    dispatch_weights = [1]

    LL.position_at_end(undef_bb)
    LL.unreachable()
    
    # all insts below. in: dispatch, dom: dispatch, entry
    for opid, insn in enumerate(insns):
        insn_bb = LL.append_basic_block(insn.__class__.__name__)
        LL.position_at_end(insn_bb)
        
        cur_operand = LL.load(LL.gep(pc, [i32(1)], inbounds=True))
        upd_pc = LL.gep(pc, [i32(2)], inbounds=True)
        upd_opcode = LL.load(upd_pc)
        
        state = InterpreterState(regs, cur_operand, ispec, smem_base)
        insn.emit(LL, state, ispec)
        dispatch_switch.add_case(opid, insn_bb)
        attrs = insn.attrs()
        if attrs & EInsnAttrs.Unlikely:
            dispatch_weights.append(1)
        else:
            dispatch_weights.append(10)
        if not (attrs & EInsnAttrs.NoReturn):
            # TODO: dynamic control flow
            LL.branch(dispatch_bb)
            opcode.add_incoming(upd_opcode, LL.block)
            pc.add_incoming(upd_pc, LL.block)
            
            for reg_b, assn_reg in zip(regs, state.assn_regs):
                reg_b.add_incoming(assn_reg, LL.block)
    dispatch_switch.set_weights(dispatch_weights)


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
