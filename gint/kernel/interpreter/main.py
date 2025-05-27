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
    LoadGlobalU8()
]

ILP = 8


def build_main_loop(LL: PlatformIRBuilder):
    # Stages for main interpreter loop:
    # issue load next instruction from memory (but don't use) (probably in L1)
    # dispatch!
    # come back from instructions directly to dispatch. phi-in local vars
    ispec = get_spec(ILP)
    
    entry_bb = LL.block
    dispatch_bb = LL.append_basic_block("dispatch")
    # back_bb = LL.append_basic_block("back")
    undef_bb = LL.append_basic_block("unreachable")
    
    LL.position_at_end(entry_bb)
    entry_pc = LL.bitcast(LL.arg(0), i32.as_pointer(1))
    # entry_insn = LL.load(code)
    # entry_next_pc = i32(0)

    state = InterpreterState([init for name, init in ispec.flat_reg_inits()], i32(0), ispec)
    # issue LoadTensorInfos() once?
    # not yet because we don't see improvements in runtime, while reg wasted
    LoadTensorInfos().emit(LL, state, ispec)
    post_entry_bb = LL.block
    
    LL.branch(dispatch_bb)
    
    LL.position_at_end(dispatch_bb)  # in: post_entry, insns
    regs = [LL.phi(init.type, name) for name, init in ispec.flat_reg_inits()]
    for reg, assn_reg in zip(regs, state.assn_regs):
        reg.add_incoming(assn_reg, post_entry_bb)
    
    # cur_insn = LL.phi(ir.VectorType(i32, 2))
    pc = LL.phi(i32.as_pointer(1))
    # cur_insn.add_incoming(entry_insn, post_entry_bb)
    pc.add_incoming(entry_pc, post_entry_bb)
    opcode = LL.load(pc)
    
    # next_insn = LL.load(LL.gep(code, [next_pc], inbounds=True))
    dispatch_switch = LL.switch(opcode, undef_bb)
    dispatch_weights = [1]

    LL.position_at_end(undef_bb)
    LL.unreachable()
    
    # all insts below. in: dispatch, dom: dispatch, entry
    for opid, insn in enumerate(insns):
        insn_bb = LL.append_basic_block(insn.__class__.__name__)
        LL.position_at_end(insn_bb)
        cur_operand = LL.load(LL.gep(pc, [i32(1)], inbounds=True))
        state = InterpreterState(regs, cur_operand, ispec)
        insn.emit(LL, state, ispec)
        dispatch_switch.add_case(opid, insn_bb)
        attrs = insn.attrs()
        if attrs & EInsnAttrs.Unlikely:
            dispatch_weights.append(1)
        else:
            dispatch_weights.append(10)
        if not (attrs & EInsnAttrs.NoReturn):
            # TODO: dynamic control flow
            upd_pc = LL.gep(pc, [i32(2)], inbounds=True)
            LL.branch(dispatch_bb)
            pc.add_incoming(upd_pc, LL.block)
            
            for reg_b, assn_reg in zip(regs, state.assn_regs):
                reg_b.add_incoming(assn_reg, LL.block)
    dispatch_switch.set_weights(dispatch_weights)


GEvalFType = ir.FunctionType(void, [
    i32.as_pointer(1),  # code
    TensorInfo.as_pointer(1),  # tensor info
    i32  # num of tensors
])


def build_interpreter_main_nvptx() -> PlatformIRBuilder:
    LL = NVPTXIRBuilder.create_kernel_module(GEvalFType, "geval")
    build_main_loop(LL)
    return LL
