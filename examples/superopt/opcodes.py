"""Search space: opcodes with stack effects for superoptimization."""

import numpy as np
from dataclasses import dataclass


def _f2i(f):
    """Float32 to int32 bit-reinterpretation."""
    return int(np.float32(f).view(np.int32))


# Opcode numbers (from gint.kernel.interpreter.main.INSNS).
# Hardcoded for speed — avoids importing the full kernel module at startup.
OP_HALT     = 0
OP_NOP      = 1
OP_FADD     = 2
OP_FMUL     = 3
OP_FSUB     = 5
OP_FRSUB    = 6
OP_FNEG     = 7
OP_FDIV     = 8
OP_FRDIV    = 9
OP_FREM     = 10
OP_FPUSH    = 11   # LoadImm
OP_FADDIMM  = 12
OP_FMULIMM  = 13
OP_LOAD_1D  = 15   # LoadGlobal1DF32
OP_STORE_1D = 16   # StoreGlobal1DF32
OP_POP      = 22
OP_DUP      = 24
OP_DUPX1    = 25
OP_DUPX2    = 26
OP_FGT      = 28
OP_FLT      = 29
OP_FGE      = 30
OP_FLE      = 31
OP_FEQ      = 32
OP_FNE      = 33
OP_SELECT   = 35
OP_FSQRT    = 40
OP_FSIN     = 41
OP_FCOS     = 42
OP_FEXP     = 49
OP_FEXP2    = 50
OP_FLOG     = 51
OP_FLOG2    = 52
OP_FRSQRT   = 53
OP_FERF     = 54
OP_SWAP     = 55
OP_FRCP     = 103

MAX_STACK = 8


@dataclass(frozen=True)
class SearchOp:
    """An instruction in the search space."""
    opcode:     int
    operand:    int
    name:       str
    min_depth:  int    # minimum stack depth to execute
    net_effect: int    # change in stack depth after execution


def build_search_ops(include_transcendental=False, immediate_values=None):
    """Build the list of search space operations.

    Returns a list of SearchOp.  Immediate-bearing opcodes are expanded into
    one SearchOp per value so the enumerator treats them as distinct atoms.
    """
    if immediate_values is None:
        immediate_values = [0.0, 0.5, 1.0, 2.0, -1.0]

    ops = [
        # --- binary arithmetic (pop 2, push 1 → net -1, min 2) ---
        SearchOp(OP_FADD,  0, "fadd",  2, -1),
        SearchOp(OP_FMUL,  0, "fmul",  2, -1),
        SearchOp(OP_FSUB,  0, "fsub",  2, -1),
        SearchOp(OP_FRSUB, 0, "frsub", 2, -1),
        SearchOp(OP_FDIV,  0, "fdiv",  2, -1),
        SearchOp(OP_FRDIV, 0, "frdiv", 2, -1),
        SearchOp(OP_FREM,  0, "frem",  2, -1),
        # --- unary arithmetic (pop 1, push 1 → net 0, min 1) ---
        SearchOp(OP_FNEG,  0, "fneg",  1, 0),
        SearchOp(OP_FRCP,  0, "frcp",  1, 0),
        # --- comparisons (pop 2, push 1 → net -1, min 2) ---
        SearchOp(OP_FGT, 0, "fgt", 2, -1),
        SearchOp(OP_FLT, 0, "flt", 2, -1),
        SearchOp(OP_FGE, 0, "fge", 2, -1),
        SearchOp(OP_FLE, 0, "fle", 2, -1),
        SearchOp(OP_FEQ, 0, "feq", 2, -1),
        SearchOp(OP_FNE, 0, "fne", 2, -1),
        # --- select (pop 3, push 1 → net -2, min 3) ---
        SearchOp(OP_SELECT, 0, "fselect", 3, -2),
        # --- stack manipulation ---
        SearchOp(OP_DUP,   0, "dup",   1,  1),
        SearchOp(OP_DUPX1, 0, "dupx1", 2,  1),
        SearchOp(OP_DUPX2, 0, "dupx2", 3,  1),
        SearchOp(OP_SWAP,  0, "swap",  2,  0),
        SearchOp(OP_POP,   0, "pop",   1, -1),
    ]

    # --- immediates: fpush(v) → net +1, min 0 ---
    for v in immediate_values:
        ops.append(SearchOp(OP_FPUSH, _f2i(v), f"fpush({v:g})", 0, 1))

    # --- faddimm(v) → net 0, min 1 (skip identity 0) ---
    for v in immediate_values:
        if v != 0.0:
            ops.append(SearchOp(OP_FADDIMM, _f2i(v), f"faddimm({v:g})", 1, 0))

    # --- fmulimm(v) → net 0, min 1 (skip identity 1 and annihilator 0) ---
    for v in immediate_values:
        if v != 1.0 and v != 0.0:
            ops.append(SearchOp(OP_FMULIMM, _f2i(v), f"fmulimm({v:g})", 1, 0))

    if include_transcendental:
        for code, name in [
            (OP_FSQRT,  "fsqrt"),
            (OP_FRSQRT, "frsqrt"),
            (OP_FEXP,   "fexp"),
            (OP_FEXP2,  "fexp2"),
            (OP_FLOG,   "flog"),
            (OP_FLOG2,  "flog2"),
            (OP_FSIN,   "fsin"),
            (OP_FCOS,   "fcos"),
            (OP_FERF,   "ferf"),
        ]:
            ops.append(SearchOp(code, 0, name, 1, 0))

    return ops
