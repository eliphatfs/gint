from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder
from .instruction import EInsnAttrs
from .instructions.load_store import LoadTensorInfos, LoadGlobalF32, StoreGlobalF32
from .instructions.control import Halt
from .state import InterpreterState, get_spec
from .structs import TensorInfo


def build_main_loop(LL: PlatformIRBuilder):
    # Stages for main interpreter loop:
    # issue load next instruction from memory (but don't use) (probably in L1)
    # dispatch!
    # come back from dispatch. phi-in local vars
    ispec = get_spec()
    
    entry_bb = LL.block
    dispatch_bb = LL.append_basic_block("dispatch")
    back_bb = LL.append_basic_block("back")
    undef_bb = LL.append_basic_block("unreachable")
    
    code = LL.arg(0)
    
    LL.position_at_end(entry_bb)
    entry_opcode = LL.load(code)
    entry_operand = LL.load(LL.gep(code, [i32(1)], inbounds=True))
    entry_next_pc = i32(2)
    LL.branch(dispatch_bb)
    
    LL.position_at_end(dispatch_bb)  # in: entry, back
    regs = [LL.phi(init.type, name) for name, init in ispec.flat_reg_inits()]
    for reg, (name, init) in zip(regs, ispec.flat_reg_inits()):
        reg.add_incoming(init, entry_bb)
    
    cur_opcode = LL.phi(i32)
    cur_operand = LL.phi(i32)
    next_pc = LL.phi(i32)
    cur_opcode.add_incoming(entry_opcode, entry_bb)
    cur_operand.add_incoming(entry_operand, entry_bb)
    next_pc.add_incoming(entry_next_pc, entry_bb)
    opcode = cur_opcode
    next_opcode = LL.load(LL.gep(code, [next_pc], inbounds=True))
    next_operand = LL.load(LL.gep(code, [LL.add(next_pc, i32(1))], inbounds=True))
    dispatch_switch = LL.switch(opcode, undef_bb)
    
    LL.position_at_end(back_bb)  # in: insts
    reg_bs = [LL.phi(phi.type, 'back_' + phi.name) for phi in regs]
    for reg, reg_b in zip(regs, reg_bs):
        reg.add_incoming(reg_b, back_bb)
    
    cur_opcode.add_incoming(next_opcode, back_bb)
    cur_operand.add_incoming(next_operand, back_bb)
    upd_pc = LL.add(next_pc, i32(2))
    next_pc.add_incoming(upd_pc, back_bb)
    LL.branch(dispatch_bb)

    LL.position_at_end(undef_bb)
    LL.unreachable()
    
    # all insts below. in: dispatch, dom: dispatch, entry
    insns = [
        Halt(),
        LoadTensorInfos(),
        LoadGlobalF32(),
        StoreGlobalF32(),
    ]
    for opid, insn in enumerate(insns):
        state = InterpreterState(regs, cur_operand, ispec)
        insn_bb = LL.append_basic_block(insn.__class__.__name__)
        LL.position_at_end(insn_bb)
        insn.emit(LL, state, ispec)
        dispatch_switch.add_case(opid, insn_bb)
        attrs = insn.attrs()
        if not (attrs & EInsnAttrs.NoReturn):
            LL.branch(back_bb)
            for reg_b, assn_reg in zip(reg_bs, state.assn_regs):
                reg_b.add_incoming(assn_reg, LL.block)  # todo: change to state variable


GEvalFType = ir.FunctionType(void, [
    i32.as_pointer(1),  # code
    TensorInfo.as_pointer(1),  # tensor info
    i32  # num of tensors
])


def build_interpreter_main_nvptx() -> PlatformIRBuilder:
    LL = NVPTXIRBuilder.create_kernel_module(GEvalFType, "geval")
    build_main_loop(LL)
    return LL
