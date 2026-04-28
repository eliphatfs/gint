"""Static analysis of gint bytecode programs.

Walks a recorded bytecode stream and reports peak stack depth and which
virtual registers are touched. Useful for kernel-variant selection (does
this program need the full pool=12/regs=8 kernel, or fit in something
smaller?), debugging stack-discipline bugs, and writing partitioner
heuristics.

The stack-effect table is hand-maintained; `tests/test_analyzer.py`
asserts every opcode in `INSNS` has an entry so it cannot drift silently.
"""
from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy

from ..kernel.interpreter.main import INSNS


# (min_depth_required, net_stack_effect) for every opcode.
# - min_depth: minimum stack depth before the instruction can execute.
# - net_effect: sp change after the instruction (push - pop).
# Halt is a terminator; the analyzer stops at it and does not apply the entry.
_EFFECTS: dict[int, tuple[int, int]] = {
    0:  (0,  0),    # Halt (terminator)
    1:  (0,  0),    # Nop
    # binary float arith
    2:  (2, -1),    # FAdd
    3:  (2, -1),    # FMul
    4:  (3, -2),    # FMA: pop 3, push 1
    5:  (2, -1),    # FSub
    6:  (2, -1),    # FRSub
    7:  (1,  0),    # FNeg
    8:  (2, -1),    # FDiv
    9:  (2, -1),    # FRDiv
    10: (2, -1),    # FRem
    # immediates / fused-imm
    11: (0,  1),    # LoadImm
    12: (1,  0),    # FAddImm
    13: (1,  0),    # FMulImm
    14: (1,  0),    # FMAImm
    # 1D global load/store
    15: (0,  1),    # LoadGlobal1DF32
    16: (1, -1),    # StoreGlobal1DF32
    17: (0,  1),    # LoadGlobal1DF16
    18: (1, -1),    # StoreGlobal1DF16
    19: (0,  1),    # LoadGlobal1DBF16
    20: (1, -1),    # StoreGlobal1DBF16
    21: (0,  1),    # LoadGlobal1DU8
    # stack manipulation
    22: (1, -1),    # Pop
    23: (2, -2),    # Pop2
    24: (1,  1),    # Dup
    25: (2,  1),    # DupX1
    26: (3,  1),    # DupX2
    27: (2,  2),    # Dup2
    # comparisons / select / approx
    28: (2, -1),    # FGt
    29: (2, -1),    # FLt
    30: (2, -1),    # FGe
    31: (2, -1),    # FLe
    32: (2, -1),    # FEq
    33: (2, -1),    # FNe
    34: (2, -1),    # FApprox
    35: (3, -2),    # FSelect
    # warp reductions (no stack change — operate in place on TOS lanes)
    36: (1,  0),    # WarpAllReduceSum
    37: (1,  0),    # WarpAllReduceMax
    38: (1,  0),    # WarpAllReduceMin
    39: (1,  0),    # WarpAllReduceProd
    # transcendentals (unary)
    40: (1,  0),    # FSqrt
    41: (1,  0),    # FSin
    42: (1,  0),    # FCos
    43: (1,  0),    # FTan
    44: (1,  0),    # FArcSin
    45: (1,  0),    # FArcCos
    46: (1,  0),    # FArcTan
    47: (2, -1),    # FArcTan2 (binary)
    48: (2, -1),    # FPow (binary)
    49: (1,  0),    # FExp
    50: (1,  0),    # FExp2
    51: (1,  0),    # FLog
    52: (1,  0),    # FLog2
    53: (1,  0),    # FRSqrt
    54: (1,  0),    # FErf
    55: (2,  0),    # Swap
    # integer arith (binary)
    56: (2, -1),    # IAdd
    57: (2, -1),    # IMul
    58: (2, -1),    # ISub
    59: (2, -1),    # IDiv
    60: (2, -1),    # IRem
    61: (2, -1),    # IShl
    62: (2, -1),    # IShr
    63: (2, -1),    # IAnd
    64: (2, -1),    # IOr
    65: (2, -1),    # IXor
    # indirect (gather/scatter)
    66: (1,  0),    # LoadGlobal1DF32Indirect: pop indices, push values
    67: (2, -2),    # StoreGlobal1DF32Indirect: pop indices and values
    # packed immediates
    68: (0,  1),    # LoadImm4F (fpush4)
    69: (0,  1),    # LoadImm4I (ipush4)
    # 2D global load/store (transpose flavor)
    70: (0,  1),    # LoadGlobal2DTF32
    71: (1, -1),    # StoreGlobal2DTF32
    72: (0,  1),    # LoadGlobal2DTF16
    73: (1, -1),    # StoreGlobal2DTF16
    74: (0,  1),    # LoadGlobal2DTBF16
    75: (1, -1),    # StoreGlobal2DTBF16
    76: (0,  1),    # LoadGlobal2DTU8
    # 2D global load/store (width flavor)
    77: (0,  1),    # LoadGlobal2DWF32
    78: (1, -1),    # StoreGlobal2DWF32
    79: (0,  1),    # LoadGlobal2DWF16
    80: (1, -1),    # StoreGlobal2DWF16
    81: (0,  1),    # LoadGlobal2DWBF16
    82: (1, -1),    # StoreGlobal2DWBF16
    83: (0,  1),    # LoadGlobal2DWU8
    # block iterator advance (in tensor info, not stack)
    84: (0,  0),    # AdvanceBlock2D
    85: (0,  0),    # AdvanceBase
    # width-lane ops
    86: (1,  1),    # DupBroadcastW
    # virtual registers (8 regs × {load, store})
    87: (0,  1), 88: (0,  1), 89: (0,  1), 90: (0,  1),
    91: (0,  1), 92: (0,  1), 93: (0,  1), 94: (0,  1),
    95: (1, -1), 96: (1, -1), 97: (1, -1), 98: (1, -1),
    99: (1, -1), 100:(1, -1), 101:(1, -1), 102:(1, -1),
    # misc
    103:(1,  0),    # FRcp
    104:(1,  0),    # FPermW
    105:(2, -1),    # FShuf2 (pop 2, push 1)
}

_HALT_OPCODE = 0
_REG_LOAD_BASE = 87   # FLoadReg0..FLoadReg7 occupy 87..94
_REG_STORE_BASE = 95  # FStoreReg0..FStoreReg7 occupy 95..102
_NUM_REGS = 8


@dataclass
class BytecodeStats:
    """Summary of a bytecode program's resource usage.

    Attributes:
        num_instructions: count of instructions executed up to (and not
            including) the first Halt, or the end of the stream if no Halt.
        max_stack: peak stack depth observed (including transient peaks
            during an instruction, modeled as max of pre- and post-state).
        max_reg_idx: highest virtual-register index touched by any
            FLoadRegN / FStoreRegN, or -1 if no register is used.
        regs_used: set of virtual-register indices touched.
        min_pool_size: minimum unified pool size that fits this program.
            Computed as the worst snapshot of stack_depth + (max_reg_idx+1)
            seen during execution; respects the aliasing layout where
            registers occupy the top of the pool.
    """
    num_instructions: int
    max_stack: int
    max_reg_idx: int
    regs_used: frozenset
    min_pool_size: int


BytecodeInput = Union[
    Sequence[Sequence[int]],   # list of [opcode, operand] pairs
    numpy.ndarray,             # int32 array shape (N, 2) or flat (2N,)
]


def _normalize(bc: BytecodeInput) -> Iterable[tuple[int, int]]:
    if isinstance(bc, numpy.ndarray):
        if bc.ndim == 1:
            assert bc.size % 2 == 0, "flat bytecode must have even length"
            it = bc.reshape(-1, 2)
        else:
            it = bc
        for row in it:
            yield int(row[0]), int(row[1])
    else:
        for row in bc:
            yield int(row[0]), int(row[1])


def analyze_bytecode(bc: BytecodeInput) -> BytecodeStats:
    """Walk a bytecode stream and compute resource usage stats.

    Accepts either a list of [opcode, operand] pairs (as produced by the
    @bytecode decorator's recorded `bc`) or a numpy int32 array of shape
    (N, 2) or flat (2N,).
    """
    depth = 0
    max_d = 0
    max_reg = -1
    regs_used: set[int] = set()
    n_insns = 0
    min_pool = 0

    for opcode, _operand in _normalize(bc):
        if opcode == _HALT_OPCODE:
            break
        n_insns += 1
        try:
            min_d, net = _EFFECTS[opcode]
        except KeyError as e:
            raise ValueError(f"unknown opcode {opcode}") from e
        if depth < min_d:
            raise ValueError(
                f"insn {n_insns - 1} (opcode {opcode}) needs depth {min_d}, "
                f"have {depth}"
            )
        if _REG_LOAD_BASE <= opcode < _REG_LOAD_BASE + _NUM_REGS:
            r = opcode - _REG_LOAD_BASE
            regs_used.add(r); max_reg = max(max_reg, r)
        elif _REG_STORE_BASE <= opcode < _REG_STORE_BASE + _NUM_REGS:
            r = opcode - _REG_STORE_BASE
            regs_used.add(r); max_reg = max(max_reg, r)
        # Peak occurs either at the pre-state or post-state (whichever is larger).
        peak = depth if net >= 0 else depth
        post = depth + net
        peak = max(peak, post)
        if peak > max_d:
            max_d = peak
        # Pool snapshot: stack peak + reserved register slots up to highest seen
        snap = peak + (max_reg + 1 if max_reg >= 0 else 0)
        if snap > min_pool:
            min_pool = snap
        depth = post

    return BytecodeStats(
        num_instructions=n_insns,
        max_stack=max_d,
        max_reg_idx=max_reg,
        regs_used=frozenset(regs_used),
        min_pool_size=min_pool,
    )


def stack_effect(opcode: int) -> tuple[int, int]:
    """Return (min_depth, net_effect) for a single opcode.

    Raises KeyError if the opcode is unknown.
    """
    return _EFFECTS[opcode]


def known_opcodes() -> frozenset:
    """Set of opcodes the analyzer has stack-effect data for."""
    return frozenset(_EFFECTS)
