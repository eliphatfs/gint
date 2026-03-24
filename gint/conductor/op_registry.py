"""
Registry mapping ATen operations to gint bytecode emission strategies.

Each entry in OP_REGISTRY describes how to lower an ATen op into one or more
stack-machine instructions.  Emission functions are called while a FrontendState
context is active so they append directly into the running bytecode list via the
frontend module.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional

import torch
from ..host import frontend as fe


class OpKind(Enum):
    ELEMENTWISE = auto()   # element-wise, same shape in/out
    REDUCTION   = auto()   # reduces elements (warp-level for now)
    METADATA    = auto()   # shape/stride-only change, identity on stack


@dataclass
class OpDescriptor:
    arity:            int                 # net stack inputs consumed by the op
    kind:             OpKind
    emit_fn:          Callable            # emit_fn(node) – called inside _frontend_state context
    # arg_order: if set, args are handled in this index order before emitting.
    # None means natural order (0, 1, …).  Useful for non-commutative ops where
    # the stack convention differs from the ATen arg order.
    arg_order:        Optional[List[int]] = None
    # check_fn: optional extra predicate – returns True when the op is supported
    # for this particular node (e.g. checking kwargs).
    check_fn:         Optional[Callable]  = None
    # Extra stack slots used *internally* at peak during emission (above the
    # arity inputs already on the stack).  The partitioner uses this to ensure
    # the hardware stack never overflows.
    peak_stack_extra: int                 = 0


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------

def _ew1(fe_fn) -> OpDescriptor:
    """Unary elementwise – wraps a single frontend call."""
    return OpDescriptor(arity=1, kind=OpKind.ELEMENTWISE, emit_fn=lambda _node: fe_fn())


def _ew2(fe_fn, arg_order=None) -> OpDescriptor:
    """Binary elementwise – wraps a single frontend call."""
    return OpDescriptor(arity=2, kind=OpKind.ELEMENTWISE, emit_fn=lambda _node: fe_fn(),
                        arg_order=arg_order)


def _meta1() -> OpDescriptor:
    """Unary metadata op – identity on stack, no bytecode emitted.

    Only arg[0] (the tensor) is pushed; shape/dim/size args are ignored.
    """
    return OpDescriptor(arity=1, kind=OpKind.METADATA,
                        emit_fn=lambda _node: None, arg_order=[0])


# ---------------------------------------------------------------------------
# Custom multi-instruction sequences
# ---------------------------------------------------------------------------

def _emit_relu(_node):
    """relu(x) = x * (x > 0).

    Stack trace (rightmost = top):
      [x]           start
      [x, x]        dup
      [x, x, 0]     fpush(0)
      [x, cond]     flt: 0 < x → cond = float(x > 0), pops 2
      [relu(x)]     fmul: x * cond = x if x>0 else 0

    Discovered by superoptimizer (was 5 insns, now 4).
    """
    fe.dup()
    fe.fpush(0.0)
    fe.flt()
    fe.fmul()


def _emit_abs(_node):
    """abs(x) = select(-x > 0, -x, x) = select(x < 0, -x, x).

    Stack trace (rightmost = top):
      [x]           start
      [x, x]        dup
      [x, -x]       fneg
      [x, -x, -x]   dup
      [abs(x)]      fselect: cond=-x, true=-x(x<0), false=x(x>=0)

    fselect treats peek(0) as float condition (>0 → true), so when -x > 0
    (i.e. x < 0) it picks peek(1) = -x, otherwise peek(2) = x.

    Discovered by superoptimizer (was 7 insns, now 4).
    """
    fe.dup()
    fe.fneg()
    fe.dup()
    fe.fselect()


def _emit_gelu(_node):
    """gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  (exact form, not tanh approx)."""
    RSQRT2 = float(np.float32(1.0 / np.sqrt(2.0)))
    # [x]
    # dup            -> [x, x]
    # fmulimm(1/√2) -> [x, x/√2]
    # ferf           -> [x, erf(x/√2)]
    # faddimm(1.0)  -> [x, 1+erf(x/√2)]
    # fmulimm(0.5)  -> [x, 0.5*(1+erf(x/√2))]
    # fmul           -> [x * 0.5*(1+erf(x/√2))] = gelu(x)  ✓
    fe.dup()
    fe.fmulimm(RSQRT2)
    fe.ferf()
    fe.faddimm(1.0)
    fe.fmulimm(0.5)
    fe.fmul()


def _check_gelu_approx(node) -> bool:
    return node.kwargs.get('approximate', 'none') == 'none'


def _emit_silu(_node):
    """silu(x) = x * sigmoid(x) = x / (1 + exp(-x))."""
    # [x]
    # dup       -> [x, x]
    # fneg      -> [x, -x]
    # fexp      -> [x, exp(-x)]
    # faddimm(1)-> [x, 1+exp(-x)]
    # frdiv     -> peek(1)/peek(0) = x/(1+exp(-x)) = silu(x)  ✓
    fe.dup()
    fe.fneg()
    fe.fexp()
    fe.faddimm(1.0)
    fe.frdiv()


def _emit_leaky_relu(node):
    """leaky_relu(x) = select(x>0, x, neg_slope*x)."""
    # negative_slope is args[1] in the AOT graph (positional), fall back to kwargs.
    neg_slope = float(node.args[1]) if len(node.args) > 1 else float(node.kwargs.get('negative_slope', 0.01))
    # [x]
    # dup              -> [x, x]
    # fmulimm(ns)     -> [x, ns*x]
    # dupx1            -> [x, ns*x, x]       (inserts x below top)
    # fselect: cond=x, true=ns*x, false=x ... wait, peek order:
    #   peek(0)=x (cond), peek(1)=ns*x (true-if-cond>0), peek(2)=x (false)
    #   when x>0: picks peek(1)=ns*x — WRONG, we want x when x>0.
    #
    # We need: select(x>0, x, ns*x).  dupx1 gives [x, ns*x, x].
    # fselect: cond=x(top), true=ns*x, false=x(bottom).
    # When x>0: true branch = ns*x.  That's leaky_relu inverted!
    # Fix: reverse the branches.  Use [ns*x, x, x] instead:
    #   dup -> fmulimm(ns) -> dupx1 gives [ns*x, fmulimm_result, x]... no.
    #
    # Actually dupx1 copies top and inserts below second:
    #   [x, ns*x] -> dupx1 -> [x, ns*x, ns*x] ... no, dupx1 copies top,
    #   inserts 1 below: [a, b] -> [a, b, a]?  Let me check.
    #
    # DupX1: v1=pop, v2=pop, push(v1), push(v2), push(v1)
    #   [x, ns*x]: v1=ns*x, v2=x -> push(ns*x), push(x), push(ns*x)
    #   = [ns*x, x, ns*x]
    # fselect: cond=ns*x, true=x, false=ns*x
    #   when ns*x>0 (i.e. x>0 for ns>0): picks x  ✓
    #   when ns*x<=0 (i.e. x<=0): picks ns*x  ✓
    # = leaky_relu(x)  ✓
    #
    # Discovered by superoptimizer (was 7 insns, now 4).
    fe.dup()
    fe.fmulimm(neg_slope)
    fe.dupx1()
    fe.fselect()


def _emit_add_tensor(node):
    """add.Tensor with optional alpha scaling: a + alpha*b."""
    alpha = node.kwargs.get('alpha', 1)
    if alpha != 1:
        fe.fmulimm(float(alpha))   # scale top (b) before adding
    fe.fadd()


def _emit_sub_tensor(node):
    """sub.Tensor with optional alpha scaling: a - alpha*b."""
    alpha = node.kwargs.get('alpha', 1)
    if alpha != 1:
        fe.fmulimm(float(alpha))
    fe.frsub()   # second - top = a - (alpha*b)


# ---------------------------------------------------------------------------
# Main registry
# ---------------------------------------------------------------------------

OP_REGISTRY: dict = {
    # --- Binary arithmetic ---
    # Note: args are pushed left-to-right so top=b, second=a.
    # FRSub = second-top = a-b; FRDiv = second/top = a/b.
    torch.ops.aten.add.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_add_tensor),
    torch.ops.aten.mul.Tensor:
        _ew2(fe.fmul),
    torch.ops.aten.sub.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_sub_tensor),
    torch.ops.aten.div.Tensor:
        _ew2(fe.frdiv),
    torch.ops.aten.remainder.Tensor:
        _ew2(fe.frem),

    # --- Comparison (produce 0.0/1.0 float) ---
    # Tensor variants: aten.gt(self, other) = self > other.
    # FGt = peek(0)>peek(1) = top>second.  With natural push [self,other] top=other:
    #   FGt = other>self  (wrong).  Reverse args so self is on top, then use FGt.
    torch.ops.aten.gt.Tensor:  _ew2(fe.fgt, arg_order=[1, 0]),
    torch.ops.aten.lt.Tensor:  _ew2(fe.flt, arg_order=[1, 0]),
    torch.ops.aten.ge.Tensor:  _ew2(fe.fge, arg_order=[1, 0]),
    torch.ops.aten.le.Tensor:  _ew2(fe.fle, arg_order=[1, 0]),
    torch.ops.aten.eq.Tensor:  _ew2(fe.feq),   # symmetric
    torch.ops.aten.ne.Tensor:  _ew2(fe.fne),   # symmetric

    # Scalar variants: aten.gt.Scalar(self, scalar) = self > scalar.
    # Natural push [self, scalar]: self at second, scalar at top.
    # Use the "flipped" instruction so top op second = scalar op self = inverse.
    # e.g. FLt = peek(0)<peek(1) = scalar<self = self>scalar ✓
    torch.ops.aten.gt.Scalar:  _ew2(fe.flt),   # scalar < self  = self > scalar
    torch.ops.aten.lt.Scalar:  _ew2(fe.fgt),   # scalar > self  = self < scalar
    torch.ops.aten.ge.Scalar:  _ew2(fe.fle),   # scalar <= self = self >= scalar
    torch.ops.aten.le.Scalar:  _ew2(fe.fge),   # scalar >= self = self <= scalar
    torch.ops.aten.eq.Scalar:  _ew2(fe.feq),   # symmetric
    torch.ops.aten.ne.Scalar:  _ew2(fe.fne),   # symmetric

    # --- Ternary ---
    # where(condition, self, other) -> Select(cond>0, self, other)
    # Select needs: peek(0)=cond, peek(1)=self(true), peek(2)=other(false).
    # We push args in natural order [cond, self, other] giving stack [cond, self, other] top=other.
    # Rearrange to [other, self, cond] via: dupx2, pop, swap.
    #   [cond, self, other]:  dupx2 → [other, cond, self, other]
    #                         pop   → [other, cond, self]
    #                         swap  → [other, self, cond]   top=cond ✓
    # Note: using natural order (arg_order=None) ensures that an internally-computed
    # condition (e.g. result of gt.Scalar) is found at depth=0 on the virtual stack
    # rather than being buried after loading the other two args.
    torch.ops.aten.where.self:
        OpDescriptor(3, OpKind.ELEMENTWISE,
                     lambda _: (fe.dupx2(), fe.pop(), fe.swap(), fe.fselect()),
                     peak_stack_extra=1),

    # --- Unary arithmetic ---
    torch.ops.aten.neg.default:   _ew1(fe.fneg),
    torch.ops.aten.abs.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_abs, peak_stack_extra=2),

    # --- Unary transcendental ---
    torch.ops.aten.sqrt.default:  _ew1(fe.fsqrt),
    torch.ops.aten.rsqrt.default: _ew1(fe.frsqrt),
    torch.ops.aten.exp.default:   _ew1(fe.fexp),
    torch.ops.aten.exp2.default:  _ew1(fe.fexp2),
    torch.ops.aten.log.default:   _ew1(fe.flog),
    torch.ops.aten.log2.default:  _ew1(fe.flog2),
    torch.ops.aten.sin.default:   _ew1(fe.fsin),
    torch.ops.aten.cos.default:   _ew1(fe.fcos),
    torch.ops.aten.tan.default:   _ew1(fe.ftan),
    torch.ops.aten.asin.default:  _ew1(fe.fasin),
    torch.ops.aten.acos.default:  _ew1(fe.facos),
    torch.ops.aten.atan.default:  _ew1(fe.fatan),
    torch.ops.aten.erf.default:   _ew1(fe.ferf),

    # --- Activation functions (multi-instruction sequences) ---
    # peak_stack_extra: max extra hardware slots used internally beyond the input already on stack.
    # Sequences were optimized by the GPU superoptimizer (examples/superopt/).
    # relu: 4 insns (was 5), peak 2 extra (dup pushes 1 slot above input)
    # abs: 4 insns (was 7), peak 2 extra (dup+fneg+dup → 3 items, but fselect consumes 3→1)
    # leaky_relu: 4 insns (was 7), peak 2 extra
    # gelu/silu: unchanged, already optimal
    torch.ops.aten.relu.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_relu, peak_stack_extra=2),
    torch.ops.aten.gelu.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_gelu,
                     check_fn=_check_gelu_approx, peak_stack_extra=1),
    torch.ops.aten.silu.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_silu, peak_stack_extra=1),
    torch.ops.aten.leaky_relu.default:
        # arg_order=[0]: only push the tensor input; negative_slope is read from node.args[1]
        # at emit time as a fmulimm immediate — do NOT push it onto the stack.
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_leaky_relu, arg_order=[0], peak_stack_extra=2),

    # --- Metadata ops (shape/stride changes, identity on stack) ---
    # These emit no bytecode.  The value on the stack is unchanged; only the
    # FX node's output shape differs, which the broadcast plan handles via
    # per-tensor ProgramTensorInfo strides.
    torch.ops.aten.view.default:         _meta1(),
    torch.ops.aten.unsqueeze.default:    _meta1(),
    torch.ops.aten.squeeze.dim:          _meta1(),
    torch.ops.aten.expand.default:       _meta1(),
    torch.ops.aten.permute.default:      _meta1(),
    torch.ops.aten.transpose.int:        _meta1(),
    torch.ops.aten.t.default:            _meta1(),
}


def get_op_descriptor(node) -> Optional[OpDescriptor]:
    """Return the OpDescriptor for *node*, or None if not supported."""
    desc = OP_REGISTRY.get(node.target)
    if desc is None:
        return None
    if desc.check_fn is not None and not desc.check_fn(node):
        return None
    return desc
