"""Reference instruction sequences to superoptimize."""

import numpy as np
from .opcodes import (
    OP_FADD, OP_FMUL, OP_FSUB, OP_FRSUB, OP_FNEG, OP_FDIV, OP_FRDIV,
    OP_FPUSH, OP_FADDIMM, OP_FMULIMM,
    OP_DUP, OP_DUPX1, OP_DUPX2, OP_SWAP,
    OP_FGT, OP_FLT, OP_FGE, OP_FLE,
    OP_SELECT, OP_FEXP, OP_FERF, OP_FRCP,
    _f2i,
)


RSQRT2 = float(np.float32(1.0 / np.sqrt(2.0)))

# Each target maps to:
#   arity: int         — number of input tensors
#   body:  list[(int, int)]  — (opcode, operand) pairs for the compute body
#   description: str   — human-readable formula
TARGETS = {
    "relu": {
        "arity": 1,
        "body": [
            (OP_DUP,    0),
            (OP_FPUSH,  _f2i(0.0)),
            (OP_DUPX2,  0),
            (OP_FLT,    0),
            (OP_SELECT, 0),
        ],
        "description": "relu(x) = select(x>0, x, 0)",
    },
    "abs": {
        "arity": 1,
        "body": [
            (OP_DUP,    0),
            (OP_FNEG,   0),
            (OP_SWAP,   0),
            (OP_DUP,    0),
            (OP_FPUSH,  _f2i(0.0)),
            (OP_FLT,    0),
            (OP_SELECT, 0),
        ],
        "description": "abs(x) = select(x>0, x, -x)",
    },
    "gelu": {
        "arity": 1,
        "body": [
            (OP_DUP,     0),
            (OP_FMULIMM, _f2i(RSQRT2)),
            (OP_FERF,    0),
            (OP_FADDIMM, _f2i(1.0)),
            (OP_FMULIMM, _f2i(0.5)),
            (OP_FMUL,    0),
        ],
        "description": "gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))",
    },
    "silu": {
        "arity": 1,
        "body": [
            (OP_DUP,     0),
            (OP_FNEG,    0),
            (OP_FEXP,    0),
            (OP_FADDIMM, _f2i(1.0)),
            (OP_FRDIV,   0),
        ],
        "description": "silu(x) = x/(1+exp(-x))",
    },
    "leaky_relu_01": {
        "arity": 1,
        "body": [
            (OP_DUP,     0),
            (OP_FMULIMM, _f2i(0.1)),
            (OP_SWAP,    0),
            (OP_DUP,     0),
            (OP_FPUSH,   _f2i(0.0)),
            (OP_FLT,     0),
            (OP_SELECT,  0),
        ],
        "description": "leaky_relu(x, 0.1) = select(x>0, x, 0.1*x)",
    },
    # Smaller targets for quick testing / verification
    "neg": {
        "arity": 1,
        "body": [(OP_FNEG, 0)],
        "description": "-x",
    },
    "square": {
        "arity": 1,
        "body": [(OP_DUP, 0), (OP_FMUL, 0)],
        "description": "x*x",
    },
    "double": {
        "arity": 1,
        "body": [(OP_DUP, 0), (OP_FADD, 0)],
        "description": "x+x = 2*x",
    },
    "add": {
        "arity": 2,
        "body": [(OP_FADD, 0)],
        "description": "a+b",
    },
    "sub": {
        "arity": 2,
        "body": [(OP_FRSUB, 0)],
        "description": "a-b (second-top)",
    },
}


def get_target(name):
    """Return target dict by name, raising KeyError if not found."""
    if name not in TARGETS:
        raise KeyError(f"Unknown target '{name}'. Available: {', '.join(TARGETS)}")
    return TARGETS[name]


def list_targets():
    """Print all available targets."""
    for name, t in TARGETS.items():
        n = len(t["body"])
        print(f"  {name:20s}  arity={t['arity']}  insns={n:2d}  {t['description']}")
