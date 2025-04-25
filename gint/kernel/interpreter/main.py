from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder
from .instruction import EInsnAttrs
from .instructions.load_store import LoadGlobal
from .instructions.control import Halt
from .state import InterpreterState, get_spec

ILP = 4


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
    entry_insn = LL.load(code)
    entry_next_pc = i32(1)
    LL.branch(dispatch_bb)
    
    LL.position_at_end(dispatch_bb)  # in: entry, back
    regs = [LL.phi(init.type, name) for name, init in ispec.flat_reg_inits()]
    for reg, (name, init) in zip(regs, ispec.flat_reg_inits()):
        reg.add_incoming(init, entry_bb)
    
    cur_insn = LL.phi(i32)
    next_pc = LL.phi(i32)
    cur_insn.add_incoming(entry_insn, entry_bb)
    next_pc.add_incoming(entry_next_pc, entry_bb)
    opcode = cur_insn
    next_insn = LL.load(LL.gep(code, [next_pc]))
    dispatch_switch = LL.switch(opcode, undef_bb)
    
    LL.position_at_end(back_bb)  # in: insts
    reg_bs = [LL.phi(phi.type, 'back_' + phi.name) for phi in regs]
    for reg, reg_b in zip(regs, reg_bs):
        reg.add_incoming(reg_b, back_bb)
    
    cur_insn.add_incoming(next_insn, back_bb)
    upd_pc = LL.add(next_pc, i32(1))
    next_pc.add_incoming(upd_pc, back_bb)
    LL.branch(dispatch_bb)

    LL.position_at_end(undef_bb)
    LL.unreachable()
    
    # all insts below. in: dispatch, dom: dispatch, entry
    insns = [
        Halt()
    ]
    for opid, insn in enumerate(insns):
        state = InterpreterState(regs, ispec)
        insn_bb = LL.append_basic_block(insn.__class__.__name__)
        LL.position_at_end(insn_bb)
        insn.emit(LL, state, ispec)
        dispatch_switch.add_case(opid, insn_bb)
        attrs = insn.attrs()
        if not (attrs & EInsnAttrs.NoReturn):
            LL.branch(back_bb)
            for reg_b, assn_reg in zip(reg_bs, state.assn_regs):
                reg_b.add_incoming(assn_reg, insn_bb)  # todo: change to state variable


def build_interpreter_main_nvptx() -> ir.Module:
    LL = NVPTXIRBuilder.create_kernel_module(ir.FunctionType(void, [i32.as_pointer(1)]), "geval")
    build_main_loop(LL)
    return LL.module
