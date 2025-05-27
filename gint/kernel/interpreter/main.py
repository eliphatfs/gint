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
    undef_bb = LL.append_basic_block("unreachable")
    
    LL.position_at_end(entry_bb)
    
    pc_block_store = LL.alloca(i32.as_pointer(1))
    entry_pc = LL.bitcast(LL.arg(0), i32.as_pointer(1))
    LL.store(entry_pc, pc_block_store)
    entry_icache = LL.load(LL.gep(entry_pc, [LL.lane_id()], inbounds=True))

    state = InterpreterState([init for name, init in ispec.flat_reg_inits()], i32(0), ispec)
    LoadTensorInfos().emit(LL, state, ispec)
    post_entry_bb = LL.block
    
    LL.branch(dispatch_bb)
    
    LL.position_at_end(dispatch_bb)  # in: post_entry, insns
    regs = [LL.phi(init.type, name) for name, init in ispec.flat_reg_inits()]
    for reg, assn_reg in zip(regs, state.assn_regs):
        reg.add_incoming(assn_reg, post_entry_bb)
    
    pc_in_pre = LL.phi(i32)
    pc_in_pre.add_incoming(i32(0), post_entry_bb)
    icache_pre = LL.phi(i32)
    icache_pre.add_incoming(entry_icache, post_entry_bb)
    
    icache_pre_block = LL.block
    with LL.if_then(LL.icmp_unsigned('>=', pc_in_pre, LL.warp_size()), likely=False):
        pc_block = LL.load(pc_block_store)
        pc_block = LL.gep(pc_block, [LL.and_(pc_in_pre, LL.xor(LL.sub(LL.warp_size(), i32(1)), i32(-1)))], inbounds=True)
        icache_upd = LL.load(LL.gep(pc_block, [LL.lane_id()], inbounds=True))
        pc_in_upd = LL.and_(pc_in_pre, LL.sub(LL.warp_size(), i32(1)))
        icache_upd_block = LL.block
        LL.store(pc_block, pc_block_store)
    pc_in = LL.phi(i32)
    pc_in.add_incoming(pc_in_upd, icache_upd_block)
    pc_in.add_incoming(pc_in_pre, icache_pre_block)
    icache = LL.phi(i32)
    icache.add_incoming(icache_upd, icache_upd_block)
    icache.add_incoming(icache_pre, icache_pre_block)
    opcode = LL.warp_broadcast_lane(icache, pc_in)
    
    dispatch_switch = LL.switch(opcode, undef_bb)
    dispatch_weights = [1]

    LL.position_at_end(undef_bb)
    LL.unreachable()
    
    # all insts below. in: dispatch, dom: dispatch, entry
    for opid, insn in enumerate(insns):
        attrs = insn.attrs()
        insn_bb = LL.append_basic_block(insn.__class__.__name__)
        LL.position_at_end(insn_bb)
        if attrs & EInsnAttrs.Operand:
            cur_operand = LL.warp_broadcast_lane(icache, LL.add(pc_in, i32(1)))  # must be in icache
        else:
            cur_operand = None
        upd_pc = LL.add(pc_in, i32(2))
        state = InterpreterState(regs, cur_operand, ispec)
        insn.emit(LL, state, ispec)
        dispatch_switch.add_case(opid, insn_bb)
        if attrs & EInsnAttrs.Unlikely:
            dispatch_weights.append(1)
        else:
            dispatch_weights.append(10)
        if not (attrs & EInsnAttrs.NoReturn):
            # TODO: dynamic control flow
            LL.branch(dispatch_bb)
            pc_in_pre.add_incoming(upd_pc, LL.block)
            icache_pre.add_incoming(icache, LL.block)
            
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
