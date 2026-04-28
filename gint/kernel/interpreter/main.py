from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder
from ..platforms.amdgcn import AMDGCNIRBuilder
from .instruction import EInsnAttrs, Instruction
from .instructions.load_tensor_infos import emit_load_tensor_infos
from .state import StackMachineState, InvalidStateException
from .structs import TensorInfo

from .instructions.arith import *
from .instructions.arith_int import *
from .instructions.control import *
from .instructions.immediate import *
from .instructions.load_store import *
from .instructions.move import *
from .instructions.predication import *
from .instructions.reduction import *
from .instructions.special import *
from .instructions.reg import *


POOL_SIZE = 12   # unified stack+register pool; stack grows up, regs down from top
NUM_REGS = 8    # reg n = pool[POOL_SIZE-1-n]; effective max stack = POOL_SIZE - NUM_REGS = 4
MAX_STACK = 8
REG_WIDTH = 4
SMEM_PER_WARP = MAX_N_TENSORS * 7 * 4


# Kernel variants. Each entry: (pool_size, num_regs, max_stack).
# The kernel symbol baked into the fatbin/HSACO is `geval_<name>`.
# Selection at host side: smallest variant whose limits cover the program.
VARIANTS: dict[str, tuple[int, int, int]] = {
    's7':  (7,  4, 7),   # small: covers all pointwise/streaming workloads
    'l12': (12, 8, 8),   # large: covers register-heavy kernels (e.g. inv4x4)
}
DEFAULT_VARIANT = 'l12'  # back-compat default for callers that don't select


def _is_invalid_reg_op(Insn, num_regs: int) -> bool:
    """Reg-load/store classes carry a `_reg_n` attribute. Variants with fewer
    registers than the global max must skip opcodes that address out-of-range
    registers — leaving them in would alias into stack slots."""
    n = getattr(Insn, '_reg_n', None)
    return n is not None and n >= num_regs


def _ensure_smem_global(LL: PlatformIRBuilder):
    """Reuse the existing dynamic_smem global if a prior variant already
    created it in this module; otherwise create it."""
    for gv in LL.module.global_values:
        if gv.name == 'dynamic_smem':
            return gv
    smem_base = ir.GlobalVariable(LL.module, ir.ArrayType(i8, 0), name='dynamic_smem', addrspace=3)
    smem_base.linkage = 'external'
    smem_base.align = 16
    return smem_base


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
    LoadGlobal1DF32: 15,
    StoreGlobal1DF32: 16,
    LoadGlobal1DF16: 17,
    StoreGlobal1DF16: 18,
    LoadGlobal1DBF16: 19,
    StoreGlobal1DBF16: 20,
    LoadGlobal1DU8: 21,
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
    Swap: 55,
    IAdd: 56,
    IMul: 57,
    ISub: 58,
    IDiv: 59,
    IRem: 60,
    IShl: 61,
    IShr: 62,
    IAnd: 63,
    IOr: 64,
    IXor: 65,
    LoadGlobal1DF32Indirect: 66,
    StoreGlobal1DF32Indirect: 67,
    LoadImm4F: 68,
    LoadImm4I: 69,
    LoadGlobal2DTF32: 70,
    StoreGlobal2DTF32: 71,
    LoadGlobal2DTF16: 72,
    StoreGlobal2DTF16: 73,
    LoadGlobal2DTBF16: 74,
    StoreGlobal2DTBF16: 75,
    LoadGlobal2DTU8: 76,
    LoadGlobal2DWF32: 77,
    StoreGlobal2DWF32: 78,
    LoadGlobal2DWF16: 79,
    StoreGlobal2DWF16: 80,
    LoadGlobal2DWBF16: 81,
    StoreGlobal2DWBF16: 82,
    LoadGlobal2DWU8: 83,
    AdvanceBlock2D: 84,
    AdvanceBase: 85,
    DupBroadcastW: 86,
    FLoadReg0: 87,
    FLoadReg1: 88,
    FLoadReg2: 89,
    FLoadReg3: 90,
    FLoadReg4: 91,
    FLoadReg5: 92,
    FLoadReg6: 93,
    FLoadReg7: 94,
    FStoreReg0: 95,
    FStoreReg1: 96,
    FStoreReg2: 97,
    FStoreReg3: 98,
    FStoreReg4: 99,
    FStoreReg5: 100,
    FStoreReg6: 101,
    FStoreReg7: 102,
    FRcp: 103,
    FPermW: 104,
    FShuf2: 105,
}


def build_main_loop(LL: PlatformIRBuilder, pool_size: int = POOL_SIZE, num_regs: int = NUM_REGS, max_stack: int = MAX_STACK):
    smem_base = _ensure_smem_global(LL)

    # early exit warps beyond user scheduling
    with LL.if_then(LL.icmp_unsigned('>=', LL.logical_program_idx(), LL.arg(3)), False):
        LL.ret_void()

    # resolve indirect pointers if flag > 0
    is_indirect = LL.icmp_signed('>', LL.arg(4), i32(0))
    idx = LL.logical_program_idx()
    resolved = {}
    with LL.if_else(is_indirect) as (then, otherwise):
        with then:
            code_table = LL.bitcast(LL.arg(0), i32.as_pointer().as_pointer())
            tinfo_table = LL.bitcast(LL.arg(1), TensorInfo.as_pointer().as_pointer())
            resolved['ind'] = (LL.load(LL.gep(code_table, [idx])), LL.load(LL.gep(tinfo_table, [idx])), LL.block)
        with otherwise:
            resolved['dir'] = (LL.bitcast(LL.arg(0), i32.as_pointer()), LL.arg(1), LL.block)
    code_ptr = LL.phi(i32.as_pointer())
    code_ptr.add_incoming(resolved['ind'][0], resolved['ind'][2])
    code_ptr.add_incoming(resolved['dir'][0], resolved['dir'][2])
    tinfo_ptr = LL.phi(TensorInfo.as_pointer())
    tinfo_ptr.add_incoming(resolved['ind'][1], resolved['ind'][2])
    tinfo_ptr.add_incoming(resolved['dir'][1], resolved['dir'][2])

    smem_base = LL.gep(smem_base, [i32(0), LL.mul(i32(SMEM_PER_WARP), LL.thread_idx_y())])

    entry_bb = LL.block
    dispatch_bbs: dict[int, ir.Block] = {}
    dispatch_states: dict[int, StackMachineState] = {}
    for i in range(max_stack + 1):
        dispatch_bbs[i] = LL.append_basic_block("dispatch.%d" % i)
        LL.position_at_end(dispatch_bbs[i])
        dispatch_states[i] = StackMachineState(LL, smem_base, pool_size, REG_WIDTH, i, num_regs, max_stack)
    undef_bb = LL.append_basic_block("unreachable")
    
    def br_state(state: StackMachineState):
        sp = state.sp
        for s, t in zip(dispatch_states[sp].flat_regs(), state.flat_regs()):
            s.add_incoming(t, LL.block)
        return LL.branch(dispatch_bbs[sp])
    
    LL.position_at_end(entry_bb)
    entry_pc = code_ptr
    entry_opcode = LL.load(entry_pc)
    state = dispatch_states[0].clone()
    state.pc = entry_pc
    state.opcode = entry_opcode
    for i in range(pool_size):
        state.pool[i] = [f32(ir.Undefined)] * REG_WIDTH
    emit_load_tensor_infos(LL, state, tinfo_ptr)
    br_state(state)

    variant_insns = [(Insn, opid) for Insn, opid in INSNS.items() if not _is_invalid_reg_op(Insn, num_regs)]

    for i in range(max_stack + 1):
        LL.position_at_end(dispatch_bbs[i])
        dispatch_switch = LL.switch(dispatch_states[i].opcode, undef_bb)
        dispatch_weights = [1]
        for Insn, opid in variant_insns:
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
    i32.as_pointer(),  # code (direct) or i32** (indirect)
    TensorInfo.as_pointer(),  # tensor info (direct) or TensorInfo** (indirect)
    i32,  # num of tensors
    i32,  # logical grid dim before dividing warps
    i32,  # flag: >0 = indirect mode
])


def variant_kernel_name(variant: str) -> str:
    """Symbol name baked into the fatbin/HSACO for a given variant."""
    return f"geval_{variant}"


def build_interpreter_main_nvptx(variants: list[str] = None) -> PlatformIRBuilder:
    """Build an NVPTX module containing one kernel per requested variant.

    Returns the IR builder of the LAST variant; the module (`LL.module`)
    contains every variant's kernel.
    """
    if variants is None:
        variants = list(VARIANTS)
    assert variants, "must request at least one variant"
    first, *rest = variants
    pool, regs, stack = VARIANTS[first]
    LL = NVPTXIRBuilder.create_kernel_module(GEvalFType, variant_kernel_name(first))
    build_main_loop(LL, pool, regs, stack)
    for v in rest:
        pool, regs, stack = VARIANTS[v]
        LL = NVPTXIRBuilder.add_kernel(LL, GEvalFType, variant_kernel_name(v))
        build_main_loop(LL, pool, regs, stack)
    return LL


def build_interpreter_main_amdgcn(gfx: str = "gfx1100", variants: list[str] = None) -> PlatformIRBuilder:
    """Build an AMDGCN module containing one kernel per requested variant."""
    if variants is None:
        variants = list(VARIANTS)
    assert variants, "must request at least one variant"
    first, *rest = variants
    pool, regs, stack = VARIANTS[first]
    LL = AMDGCNIRBuilder.create_kernel_module(GEvalFType, variant_kernel_name(first), gfx=gfx)
    build_main_loop(LL, pool, regs, stack)
    for v in rest:
        pool, regs, stack = VARIANTS[v]
        LL = AMDGCNIRBuilder.add_kernel(LL, GEvalFType, variant_kernel_name(v))
        build_main_loop(LL, pool, regs, stack)
    return LL
