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
    # the stack convention differs from the ATen arg order.  May also be a
    # callable(node) → list[int] for descriptors that need to choose at emit
    # time (e.g. binary ops that fold a scalar second arg into an immediate
    # instruction and therefore want to skip the scalar push).
    arg_order:        Optional[object]    = None
    # check_fn: optional extra predicate – returns True when the op is supported
    # for this particular node (e.g. checking kwargs).
    check_fn:         Optional[Callable]  = None
    # Extra stack slots used *internally* at peak during emission (above the
    # arity inputs already on the stack).  The partitioner uses this to ensure
    # the hardware stack never overflows.
    peak_stack_extra: int                 = 0
    # Reduction-only: emits the pairwise combine for chunk accumulation
    # (e.g. fadd for sum, fmul for prod, max(a,b) sequence for amax).
    # Stack contract: [a, b] → [combine(a, b)].
    combine_fn:       Optional[Callable]  = None
    # Reduction-only: emits the warp-allreduce primitive (1 insn) — the
    # 32-thread reduction. Distinct from combine_fn because it's a single
    # opcode rather than a pairwise stack op.
    warp_reduce_fn:   Optional[Callable]  = None
    # Reduction-only: optional final fix-up that runs once after the
    # warp/width reduction (e.g. mean's `* 1/N`). Takes the FX node so it
    # can read shape-dependent constants. None = no post step.
    post_reduce_fn:   Optional[Callable]  = None
    # When True, the op's result is invariant under arg permutation
    # (e.g. ``a*b == b*a``).  Lets ``StackCodegen`` skip Swap-pair work
    # when the FX-declared arg order would force a stack reshuffle that
    # the op doesn't actually need.  Only meaningful for arity==2 ops
    # whose emit_fn is symmetric in its stack inputs.
    commutative:      bool                = False


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


def _is_scalar(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _arg_order_scalar_fold(node):
    """Push only args[0] when args[1] is a Python scalar so the emit_fn can
    fold it into an immediate instruction.  Otherwise push both args."""
    return [0] if _is_scalar(node.args[1]) else [0, 1]


def _emit_add_tensor(node):
    """add.Tensor / add.Scalar: a + alpha*b.

    Scalar fold: when b is a Python scalar (the common case in FX graphs
    for ``x + 1.0`` and friends), emit a single ``faddimm(alpha*b)`` —
    one instruction in place of fpush + fmulimm + fadd.
    """
    alpha = node.kwargs.get('alpha', 1)
    other = node.args[1]
    if _is_scalar(other):
        fe.faddimm(float(alpha) * float(other))
    else:
        if alpha != 1:
            fe.fmulimm(float(alpha))
        fe.fadd()


def _emit_sub_tensor(node):
    """sub.Tensor / sub.Scalar: a - alpha*b.

    Scalar fold: emit ``faddimm(-alpha*b)`` when b is a Python scalar.
    """
    alpha = node.kwargs.get('alpha', 1)
    other = node.args[1]
    if _is_scalar(other):
        fe.faddimm(-float(alpha) * float(other))
    else:
        if alpha != 1:
            fe.fmulimm(float(alpha))
        fe.frsub()   # second - top = a - (alpha*b)


def _emit_mul_binary(node):
    """mul.Tensor / mul.Scalar.  Scalar fold: ``fmulimm(b)``."""
    other = node.args[1]
    if _is_scalar(other):
        fe.fmulimm(float(other))
    else:
        fe.fmul()


def _emit_div_binary(node):
    """div.Tensor / div.Scalar: a / b.  Scalar fold: ``fmulimm(1/b)``.

    For b=0 this collapses to ``fmulimm(inf)`` which gives the same IEEE
    result as ``a / 0`` (±inf for nonzero a, NaN for zero a).
    """
    other = node.args[1]
    if _is_scalar(other):
        fe.fmulimm(1.0 / float(other))
    else:
        fe.frdiv()


def _emit_rsub_scalar(node):
    """rsub.Scalar(self, other, alpha=1) = other - alpha*self.
    Folds to ``fmulimm(-alpha); faddimm(other)`` — 2 insns vs 3 in the
    naive ``fpush + frsub`` lowering.
    """
    other = float(node.args[1])
    alpha = float(node.kwargs.get('alpha', 1))
    fe.fmulimm(-alpha)
    fe.faddimm(other)


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------

def _check_reduction_feasible(node) -> bool:
    """Innermost-dim reduction, single dim only.

    Accepts both the ``dim_IntList`` style (``dims`` is a list, e.g.
    ``aten.sum.dim_IntList``, ``aten.amax.default``) and the ``dim_int``
    style (``dim`` is a single int, e.g. ``aten.prod.dim_int``).
    """
    shape = node.args[0].meta['tensor_meta'].shape
    dims = node.args[1]
    if isinstance(dims, int):
        dim = dims % len(shape)
    else:
        if len(dims) != 1:
            return False
        dim = dims[0] % len(shape)
    return dim == len(shape) - 1  # innermost only


def _reduction_chunk_size(N: int) -> int:
    """Number of reduction elements covered per warp-load chunk.

    Mirrors the tile dispatch in conductor.compiler._select_reduction_tiling.
    Kept here so feasibility checks can compute the OOB-padding alignment
    constraint without importing the conductor-internal selector.
    """
    if N >= 128:
        return 128
    if N >= 16:
        return 32
    return 4


def _check_reduction_feasible_clean_chunks(node) -> bool:
    """As above, but additionally require N to be a multiple of the chunk size.

    Sum/mean tolerate the kernel's 0.0 OOB padding because adding zero is a
    no-op.  Prod (multiplies by zero), amax (max-with-zero clamps negative
    values), and amin (min-with-zero clamps positive values) do NOT tolerate
    it.  Until the kernel grows per-op OOB defaults, restrict these to
    reduction sizes that align with the chosen tile's chunk size.
    """
    if not _check_reduction_feasible(node):
        return False
    shape = node.args[0].meta['tensor_meta'].shape
    return shape[-1] % _reduction_chunk_size(shape[-1]) == 0


def _emit_max_pair(_node=None):
    """[a, b] → [max(a, b)]. Uses fselect(cond=b>a, true=b, false=a)."""
    fe.dup2()    # [a, b, a, b]
    fe.fgt()     # peek(0)>peek(1) = b>a → cond. Stack: [a, b, cond]
    fe.fselect() # cond>0 picks peek(1)=b, else peek(2)=a → max(a,b)


def _emit_min_pair(_node=None):
    """[a, b] → [min(a, b)]. Uses fselect(cond=b<a, true=b, false=a)."""
    fe.dup2()
    fe.flt()     # peek(0)<peek(1) = b<a
    fe.fselect()


def _emit_width_combine(combine_fn):
    """Width-lane reduction: 4-wide vector → 1 (broadcast back to all 4 lanes).

    [v0,v1,v2,v3] → dup → fperm_w(2,3,0,1) → combine
                  → [v0+v2, v1+v3, v0+v2, v1+v3] (for fadd)
                  → dup → fperm_w(1,0,3,2) → combine
                  → all 4 lanes hold sum.
    """
    fe.dup()
    fe.fperm_w(2, 3, 0, 1)
    combine_fn()
    fe.dup()
    fe.fperm_w(1, 0, 3, 2)
    combine_fn()


def _emit_mean_post(node):
    """Final * 1/N for mean."""
    shape = node.args[0].meta['tensor_meta'].shape
    fe.fmulimm(1.0 / float(shape[-1]))


def _emit_sum(_node):
    """Full reduction: warp_allreduce + width-lane combine."""
    fe.warp_allreduce_fsum()
    _emit_width_combine(fe.fadd)


def _emit_mean(node):
    """Sum + divide by count."""
    _emit_sum(node)
    _emit_mean_post(node)


def _emit_prod(_node):
    """warp_allreduce_fprod + width-lane combine via fmul."""
    fe.warp_allreduce_fprod()
    _emit_width_combine(fe.fmul)


def _emit_amax(_node):
    """warp_allreduce_fmax + width-lane combine via pairwise max."""
    fe.warp_allreduce_fmax()
    _emit_width_combine(_emit_max_pair)


def _emit_amin(_node):
    """warp_allreduce_fmin + width-lane combine via pairwise min."""
    fe.warp_allreduce_fmin()
    _emit_width_combine(_emit_min_pair)


# ---------------------------------------------------------------------------
# Composed pointwise: emitted as short instruction sequences over existing ops
# ---------------------------------------------------------------------------

def _emit_tanh(_node):
    """tanh(x) = 1 - 2 / (exp(2x) + 1).  Numerically OK at both tails.

    [x] → fmulimm(2) → [2x] → fexp → [e^(2x)] → faddimm(1) → [e^(2x)+1]
        → frcp → [1/(e^(2x)+1)] → fmulimm(-2) → faddimm(1) = tanh(x)
    """
    fe.fmulimm(2.0)
    fe.fexp()
    fe.faddimm(1.0)
    fe.frcp()
    fe.fmulimm(-2.0)
    fe.faddimm(1.0)


def _emit_sigmoid(_node):
    """sigmoid(x) = 1 / (1 + exp(-x))."""
    fe.fneg()
    fe.fexp()
    fe.faddimm(1.0)
    fe.frcp()


def _emit_square(_node):
    """square(x) = x * x."""
    fe.dup()
    fe.fmul()


def _emit_log1p(_node):
    fe.faddimm(1.0)
    fe.flog()


def _emit_expm1(_node):
    fe.fexp()
    fe.faddimm(-1.0)


def _emit_log10(_node):
    fe.flog()
    fe.fmulimm(float(np.float32(1.0 / np.log(10.0))))


def _emit_sinh(_node):
    """sinh(x) = (exp(x) - exp(-x)) / 2."""
    # [x] → dup → [x, x] → fneg → [x, -x] → fexp → [x, e^-x]
    #   → swap → [e^-x, x] → fexp → [e^-x, e^x]
    #   → fsub: top - second = e^x - e^-x → fmulimm(0.5)
    fe.dup()
    fe.fneg()
    fe.fexp()
    fe.swap()
    fe.fexp()
    fe.fsub()
    fe.fmulimm(0.5)


def _emit_cosh(_node):
    """cosh(x) = (exp(x) + exp(-x)) / 2."""
    fe.dup()
    fe.fneg()
    fe.fexp()
    fe.swap()
    fe.fexp()
    fe.fadd()
    fe.fmulimm(0.5)


def _emit_atanh(_node):
    """atanh(x) = 0.5 * log((1 + x) / (1 - x))."""
    # [x] → dup → [x, x] → faddimm(1) → [x, 1+x]
    #   → swap → [1+x, x] → fmulimm(-1) → faddimm(1) → [1+x, 1-x]
    #   → frdiv: peek(1)/peek(0) = (1+x)/(1-x) → flog → fmulimm(0.5)
    fe.dup()
    fe.faddimm(1.0)
    fe.swap()
    fe.fmulimm(-1.0)
    fe.faddimm(1.0)
    fe.frdiv()
    fe.flog()
    fe.fmulimm(0.5)


def _emit_asinh(_node):
    """asinh(x) = log(x + sqrt(x^2 + 1))."""
    # [x] → dup → dup → fmul → faddimm(1) → fsqrt → [x, sqrt(x^2+1)]
    #   → fadd → [x + sqrt(x^2+1)] → flog
    fe.dup()
    fe.dup()
    fe.fmul()
    fe.faddimm(1.0)
    fe.fsqrt()
    fe.fadd()
    fe.flog()


def _emit_acosh(_node):
    """acosh(x) = log(x + sqrt(x^2 - 1)).  Valid for x >= 1."""
    fe.dup()
    fe.dup()
    fe.fmul()
    fe.faddimm(-1.0)
    fe.fsqrt()
    fe.fadd()
    fe.flog()


# ---------------------------------------------------------------------------
# Clamping and pairwise min/max
# ---------------------------------------------------------------------------

def _emit_minimum(_node):
    """minimum(a, b): elementwise min.  Stack [a, b] → [min(a, b)]."""
    _emit_min_pair()


def _emit_maximum(_node):
    """maximum(a, b): elementwise max."""
    _emit_max_pair()


def _emit_clamp_min_scalar(node):
    """clamp_min(x, m) with scalar m.  m is on stack as the second arg."""
    _emit_max_pair()


def _emit_clamp_max_scalar(node):
    """clamp_max(x, m) with scalar m."""
    _emit_min_pair()


def _emit_clamp(node):
    """clamp(x, min=None, max=None).  arg_order=[0] – min/max are pulled
    from the FX node and emitted as immediates so we can handle the
    optional-None case without conditional pushes."""
    min_val = node.args[1] if len(node.args) > 1 else node.kwargs.get('min', None)
    max_val = node.args[2] if len(node.args) > 2 else node.kwargs.get('max', None)
    if min_val is not None:
        fe.fpush(float(min_val))
        _emit_max_pair()
    if max_val is not None:
        fe.fpush(float(max_val))
        _emit_min_pair()


def _check_clamp(node) -> bool:
    """Reject clamp with both bounds None (the partitioner would treat it
    as identity, which is fine, but the FX graph normally won't produce
    that anyway)."""
    min_val = node.args[1] if len(node.args) > 1 else node.kwargs.get('min', None)
    max_val = node.args[2] if len(node.args) > 2 else node.kwargs.get('max', None)
    return (min_val is not None) or (max_val is not None)


def _emit_hardtanh(node):
    """hardtanh(x, min_val=-1, max_val=1) = clamp(x, min_val, max_val)."""
    min_val = float(node.args[1]) if len(node.args) > 1 else -1.0
    max_val = float(node.args[2]) if len(node.args) > 2 else 1.0
    fe.fpush(min_val)
    _emit_max_pair()
    fe.fpush(max_val)
    _emit_min_pair()


def _emit_relu6(_node):
    """relu6(x) = clamp(x, 0, 6)."""
    fe.fpush(0.0)
    _emit_max_pair()
    fe.fpush(6.0)
    _emit_min_pair()


def _emit_hardsigmoid(_node):
    """hardsigmoid(x) = clamp((x + 3) / 6, 0, 1) = clamp(x/6 + 0.5, 0, 1).

    Uses fmulimm + faddimm (f32 immediates) rather than fmaimm (fp16)
    so the output stays within the eager tolerance.
    """
    fe.fmulimm(1.0 / 6.0)
    fe.faddimm(0.5)
    fe.fpush(0.0)
    _emit_max_pair()
    fe.fpush(1.0)
    _emit_min_pair()


def _emit_hardswish(_node):
    """hardswish(x) = x * clamp(x/6 + 0.5, 0, 1)."""
    fe.dup()
    fe.fmulimm(1.0 / 6.0)
    fe.faddimm(0.5)
    fe.fpush(0.0)
    _emit_max_pair()
    fe.fpush(1.0)
    _emit_min_pair()
    fe.fmul()


def _emit_softplus(_node):
    """softplus(x) = log(1 + exp(x)).  Default beta=1; assumes x is in a
    range where exp(x) doesn't overflow.  PyTorch's threshold-fallback
    isn't replicated here."""
    fe.fexp()
    fe.faddimm(1.0)
    fe.flog()


def _check_softplus(node) -> bool:
    beta = node.kwargs.get('beta', 1)
    if len(node.args) > 1:
        beta = node.args[1]
    return float(beta) == 1.0


def _emit_mish(_node):
    """mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))."""
    # [x] → dup → [x, x] → softplus → [x, sp] → tanh → [x, tanh(sp)] → fmul
    fe.dup()
    _emit_softplus(None)
    _emit_tanh(None)
    fe.fmul()


def _emit_elu(node):
    """elu(self, alpha=1, scale=1, input_scale=1).

    PyTorch defines elu = scale * (max(0, x*is) + min(0, alpha*(exp(x*is)-1))).
    Builds [false=alpha*(e^(x*is)-1), true=x*is, cond=x*is] → fselect → fmulimm(scale).
    SELU is decomposed by torch into elu with non-default alpha+scale; we honour both.
    """
    args = node.args
    alpha = float(args[1]) if len(args) > 1 else float(node.kwargs.get('alpha', 1.0))
    scale = float(args[2]) if len(args) > 2 else float(node.kwargs.get('scale', 1.0))
    input_scale = float(args[3]) if len(args) > 3 else float(node.kwargs.get('input_scale', 1.0))
    if input_scale != 1.0:
        fe.fmulimm(input_scale)
    fe.dup()
    fe.dup()
    fe.fexp()
    fe.faddimm(-1.0)
    fe.fmulimm(alpha)
    fe.dupx2()
    fe.pop()
    fe.fselect()
    if scale != 1.0:
        fe.fmulimm(scale)


def _emit_selu(_node):
    """Direct SELU lowering — kept for the rare call site that doesn't get
    decomposed into ``aten.elu.default``."""
    SELU_ALPHA = 1.6732632423543772
    SELU_SCALE = 1.0507009873554805
    fake_node = type('N', (), {'args': (None, SELU_ALPHA, SELU_SCALE), 'kwargs': {}})
    _emit_elu(fake_node)


def _emit_threshold(node):
    """threshold(x, threshold, value) = x if x > threshold else value."""
    th = float(node.args[1])
    val = float(node.args[2])
    # Want: select(x > threshold, x, value).  fselect: cond=x>threshold? Use shifted.
    # [x] → dup → [x, x] → faddimm(-th) → [x, x-th]   (cond>0 iff x>th, when th>0 the
    #   sign matches; for general th, x-th>0 ↔ x>th ✓)
    # → fpush(val) → [x, x-th, val] → swap → [x, val, x-th]
    # Want stack [false=val, true=x, cond=x-th].  Currently [x, val, x-th]: third=x ✓ (true),
    # second=val ✓ (false), top=x-th ✓ (cond).  But fselect needs false=peek(2), true=peek(1).
    # peek(2)=x (third), peek(1)=val (second), peek(0)=x-th.  So fselect picks val if
    # cond>0 (= x>th), x otherwise.  WRONG — picks val when x>th, but we want x when x>th.
    # Swap branches: build [val, x, x-th] instead.
    # [x] → dup → [x, x] → fpush(val) → [x, x, val] → swap → [x, val, x] → faddimm(-th)
    #   → [x, val, x-th].  Same as above — third=x, second=val, top=x-th.  fselect picks
    #   peek(1)=val if peek(0)>0 (= x>th).  Still wrong sense.
    # Right way: stack [false, true, cond] where false=val, true=x, cond=x-th.
    # That maps to peek(2)=val, peek(1)=x, peek(0)=x-th.  Build:
    #   [x] → fpush(val) → swap → [val, x] → dup → [val, x, x] → faddimm(-th)
    #     → [val, x, x-th].  peek(2)=val ✓, peek(1)=x ✓, peek(0)=x-th ✓.  fselect ✓.
    fe.fpush(val)
    fe.swap()
    fe.dup()
    fe.faddimm(-th)
    fe.fselect()



def _emit_addcmul(node):
    """addcmul(self, t1, t2, *, value=1) = self + value * t1 * t2.
    Stack on entry: [self, t1, t2].  Output: [self + value*t1*t2]."""
    value = float(node.kwargs.get('value', 1))
    fe.fmul()
    if value != 1.0:
        fe.fmulimm(value)
    fe.fadd()


def _emit_addcdiv(node):
    """addcdiv(self, t1, t2, *, value=1) = self + value * t1 / t2.
    Stack on entry: [self, t1, t2].  Output: [self + value*t1/t2]."""
    value = float(node.kwargs.get('value', 1))
    fe.frdiv()  # peek(1)/peek(0) = t1/t2
    if value != 1.0:
        fe.fmulimm(value)
    fe.fadd()


def _emit_lerp_scalar(node):
    """lerp.Scalar(self, end, weight) = self*(1-w) + end*w.

    Stack: [self, end] → fmulimm(w); swap; fmulimm(1-w); fadd.
    4 insns; equivalent to self + w*(end-self) but avoids dup'ing self.
    """
    w = float(node.args[2]) if len(node.args) > 2 else float(node.kwargs.get('weight'))
    fe.fmulimm(w)
    fe.swap()
    fe.fmulimm(1.0 - w)
    fe.fadd()


def _emit_hardshrink(node):
    """hardshrink(x, lambd=0.5) = x if |x| > lambd else 0."""
    lambd = float(node.args[1]) if len(node.args) > 1 else float(node.kwargs.get('lambd', 0.5))
    # Want: select(|x| > lambd, x, 0).  We need [false=0, true=x, cond=|x|-lambd].
    # [x] → dup → [x, x] → dup → [x, x, x] → fneg → [x, x, -x] → dup → [x, x, -x, -x]
    #   → fselect → [x, |x|]   (per _emit_abs)
    # → faddimm(-lambd) → [x, |x|-lambd]
    # → fpush(0) → [x, |x|-lambd, 0] → swap → [x, 0, |x|-lambd]
    # Stack: peek(2)=x (false?), peek(1)=0 (true?), peek(0)=|x|-lambd (cond).
    # fselect picks peek(1)=0 if cond>0; we want x when cond>0.  Reversed.
    # Reorder to [0, x, cond]: starting from [x] need to build [0, x, |x|-lambd].
    # [x] → fpush(0) → swap → [0, x] → dup → dup → fneg → dup → fselect → [0, x, |x|]
    #   → faddimm(-lambd) → [0, x, |x|-lambd] ✓ → fselect picks peek(1)=x when cond>0 ✓.
    fe.fpush(0.0)
    fe.swap()
    fe.dup()
    fe.dup()
    fe.fneg()
    fe.dup()
    fe.fselect()
    fe.faddimm(-lambd)
    fe.fselect()


# ---------------------------------------------------------------------------
# Main registry
# ---------------------------------------------------------------------------

OP_REGISTRY: dict = {
    # --- Binary arithmetic ---
    # Note: args are pushed left-to-right so top=b, second=a (when b is a tensor).
    # FRSub = second-top = a-b; FRDiv = second/top = a/b.
    # When b is a Python scalar, the descriptors fold it into a single
    # immediate-form instruction (faddimm/fmulimm) and arg_order skips the
    # would-be fpush so codegen stays consistent.
    torch.ops.aten.add.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_add_tensor,
                     arg_order=_arg_order_scalar_fold, commutative=True),
    torch.ops.aten.mul.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_mul_binary,
                     arg_order=_arg_order_scalar_fold, commutative=True),
    torch.ops.aten.sub.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_sub_tensor,
                     arg_order=_arg_order_scalar_fold),
    torch.ops.aten.div.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_div_binary,
                     arg_order=_arg_order_scalar_fold),
    torch.ops.aten.remainder.Tensor:
        _ew2(fe.frem),

    # Scalar variants: self is a tensor, other is a Python scalar.  Reuse
    # the same emitters; the dynamic arg_order folds the scalar in all cases.
    torch.ops.aten.add.Scalar:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_add_tensor,
                     arg_order=_arg_order_scalar_fold),
    torch.ops.aten.sub.Scalar:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_sub_tensor,
                     arg_order=_arg_order_scalar_fold),
    torch.ops.aten.mul.Scalar:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_mul_binary,
                     arg_order=_arg_order_scalar_fold),
    torch.ops.aten.div.Scalar:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_div_binary,
                     arg_order=_arg_order_scalar_fold),
    # rsub.Scalar(self, other) = other - self.  arg_order=[0]: only push self;
    # the scalar `other` is read from the FX node and emitted as faddimm.
    torch.ops.aten.rsub.Scalar:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_rsub_scalar, arg_order=[0]),

    # --- Comparison (produce 0.0/1.0 float) ---
    # Tensor variants: aten.gt(self, other) = self > other.
    # FGt = peek(0)>peek(1) = top>second.  With natural push [self,other] top=other:
    #   FGt = other>self  (wrong).  Reverse args so self is on top, then use FGt.
    torch.ops.aten.gt.Tensor:  _ew2(fe.fgt, arg_order=[1, 0]),
    torch.ops.aten.lt.Tensor:  _ew2(fe.flt, arg_order=[1, 0]),
    torch.ops.aten.ge.Tensor:  _ew2(fe.fge, arg_order=[1, 0]),
    torch.ops.aten.le.Tensor:  _ew2(fe.fle, arg_order=[1, 0]),
    torch.ops.aten.eq.Tensor:  OpDescriptor(2, OpKind.ELEMENTWISE,
                                            lambda _n: fe.feq(), commutative=True),
    torch.ops.aten.ne.Tensor:  OpDescriptor(2, OpKind.ELEMENTWISE,
                                            lambda _n: fe.fne(), commutative=True),

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
    torch.ops.aten.sqrt.default:       _ew1(fe.fsqrt),
    torch.ops.aten.rsqrt.default:      _ew1(fe.frsqrt),
    torch.ops.aten.exp.default:        _ew1(fe.fexp),
    torch.ops.aten.exp2.default:       _ew1(fe.fexp2),
    torch.ops.aten.log.default:        _ew1(fe.flog),
    torch.ops.aten.log2.default:       _ew1(fe.flog2),
    torch.ops.aten.sin.default:        _ew1(fe.fsin),
    torch.ops.aten.cos.default:        _ew1(fe.fcos),
    torch.ops.aten.tan.default:        _ew1(fe.ftan),
    torch.ops.aten.asin.default:       _ew1(fe.fasin),
    torch.ops.aten.acos.default:       _ew1(fe.facos),
    torch.ops.aten.atan.default:       _ew1(fe.fatan),
    torch.ops.aten.erf.default:        _ew1(fe.ferf),
    torch.ops.aten.reciprocal.default: _ew1(fe.frcp),
    torch.ops.aten.atan2.default:      _ew2(fe.fatan2, arg_order=[1, 0]),

    # --- Power ---
    # FPow computes top^second.  ATen arg order is (base, exponent), so push
    # exponent first → stack [exponent, base] → FPow = base^exponent.
    torch.ops.aten.pow.Tensor_Scalar: _ew2(fe.fpow, arg_order=[1, 0]),
    torch.ops.aten.pow.Tensor_Tensor: _ew2(fe.fpow, arg_order=[1, 0]),

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

    torch.ops.aten.tanh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_tanh, peak_stack_extra=1),
    torch.ops.aten.sigmoid.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_sigmoid, peak_stack_extra=1),

    # --- Composed unary math ---
    torch.ops.aten.square.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_square, peak_stack_extra=1),
    torch.ops.aten.log1p.default:
        _ew1(lambda: (fe.faddimm(1.0), fe.flog())),
    torch.ops.aten.expm1.default:
        _ew1(lambda: (fe.fexp(), fe.faddimm(-1.0))),
    torch.ops.aten.log10.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_log10),
    torch.ops.aten.sinh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_sinh, peak_stack_extra=1),
    torch.ops.aten.cosh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_cosh, peak_stack_extra=1),
    torch.ops.aten.atanh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_atanh, peak_stack_extra=1),
    torch.ops.aten.asinh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_asinh, peak_stack_extra=1),
    torch.ops.aten.acosh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_acosh, peak_stack_extra=1),

    # --- Pairwise min/max + clamping ---
    # max(a,b)/min(a,b) via dup2; fgt/flt; fselect (3 insns each).
    torch.ops.aten.maximum.default:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_maximum,
                     peak_stack_extra=2, commutative=True),
    torch.ops.aten.minimum.default:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_minimum,
                     peak_stack_extra=2, commutative=True),
    # Tensor-tensor clamp_min/max share the maximum/minimum sequence.
    torch.ops.aten.clamp_min.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_clamp_min_scalar, peak_stack_extra=2),
    torch.ops.aten.clamp_max.Tensor:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_clamp_max_scalar, peak_stack_extra=2),
    # Scalar variants: the bound is pushed as a scalar before the pair op.
    torch.ops.aten.clamp_min.default:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_clamp_min_scalar, peak_stack_extra=2),
    torch.ops.aten.clamp_max.default:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_clamp_max_scalar, peak_stack_extra=2),
    torch.ops.aten.clamp.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_clamp,
                     arg_order=[0], check_fn=_check_clamp, peak_stack_extra=3),

    # --- Higher-level activations (composed) ---
    torch.ops.aten.hardtanh.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_hardtanh, arg_order=[0], peak_stack_extra=3),
    torch.ops.aten.relu6.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_relu6, peak_stack_extra=3),
    torch.ops.aten.hardsigmoid.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_hardsigmoid, peak_stack_extra=3),
    torch.ops.aten.hardswish.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_hardswish, peak_stack_extra=4),
    torch.ops.aten.softplus.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_softplus,
                     arg_order=[0], check_fn=_check_softplus, peak_stack_extra=1),
    torch.ops.aten.mish.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_mish, peak_stack_extra=2),
    torch.ops.aten.elu.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_elu, arg_order=[0], peak_stack_extra=3),
    torch.ops.aten.selu.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_selu, peak_stack_extra=3),
    torch.ops.aten.threshold.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_threshold, arg_order=[0], peak_stack_extra=2),
    torch.ops.aten.hardshrink.default:
        OpDescriptor(1, OpKind.ELEMENTWISE, _emit_hardshrink, arg_order=[0], peak_stack_extra=4),

    # --- Composite scalar ops ---
    # addcmul: arity 3 (self, t1, t2); value is a kwarg.  Stack [self, t1, t2] → fmul → [self, t1*t2]
    # → optional fmulimm(value) → fadd.
    torch.ops.aten.addcmul.default:
        OpDescriptor(3, OpKind.ELEMENTWISE, _emit_addcmul),
    torch.ops.aten.addcdiv.default:
        OpDescriptor(3, OpKind.ELEMENTWISE, _emit_addcdiv),
    # lerp.Scalar: arity 2 (self, end); weight is a scalar arg pulled at emit time.
    torch.ops.aten.lerp.Scalar:
        OpDescriptor(2, OpKind.ELEMENTWISE, _emit_lerp_scalar, arg_order=[0, 1]),

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
    torch.ops.aten.slice.Tensor:         _meta1(),

    # --- Reduction ops ---
    # combine_fn is the pairwise op used to accumulate chunks before the
    # warp-level reduce.  For sum/mean it is fadd; for prod fmul; for
    # amax/amin it is the 3-instruction fselect-based pair sequence.
    torch.ops.aten.sum.dim_IntList:
        OpDescriptor(1, OpKind.REDUCTION, _emit_sum,
                     arg_order=[0], check_fn=_check_reduction_feasible,
                     peak_stack_extra=2, combine_fn=fe.fadd,
                     warp_reduce_fn=fe.warp_allreduce_fsum),
    torch.ops.aten.mean.dim:
        OpDescriptor(1, OpKind.REDUCTION, _emit_mean,
                     arg_order=[0], check_fn=_check_reduction_feasible,
                     peak_stack_extra=2, combine_fn=fe.fadd,
                     warp_reduce_fn=fe.warp_allreduce_fsum,
                     post_reduce_fn=_emit_mean_post),
    torch.ops.aten.prod.dim_int:
        OpDescriptor(1, OpKind.REDUCTION, _emit_prod,
                     arg_order=[0],
                     check_fn=_check_reduction_feasible_clean_chunks,
                     peak_stack_extra=2, combine_fn=fe.fmul,
                     warp_reduce_fn=fe.warp_allreduce_fprod),
    torch.ops.aten.amax.default:
        OpDescriptor(1, OpKind.REDUCTION, _emit_amax,
                     arg_order=[0],
                     check_fn=_check_reduction_feasible_clean_chunks,
                     peak_stack_extra=3, combine_fn=_emit_max_pair,
                     warp_reduce_fn=fe.warp_allreduce_fmax),
    torch.ops.aten.amin.default:
        OpDescriptor(1, OpKind.REDUCTION, _emit_amin,
                     arg_order=[0],
                     check_fn=_check_reduction_feasible_clean_chunks,
                     peak_stack_extra=3, combine_fn=_emit_min_pair,
                     warp_reduce_fn=fe.warp_allreduce_fmin),
}


def get_op_descriptor(node) -> Optional[OpDescriptor]:
    """Return the OpDescriptor for *node*, or None if not supported."""
    desc = OP_REGISTRY.get(node.target)
    if desc is None:
        return None
    if desc.check_fn is not None and not desc.check_fn(node):
        return None
    return desc


def resolve_arg_order(desc: OpDescriptor, node) -> List[int]:
    """Resolve the descriptor's ``arg_order`` for this node.

    ``arg_order`` may be a static list, a callable(node) → list[int], or
    ``None`` (natural order).  Centralising this lets the codegen and the
    fused-reduction phases share one resolution rule.
    """
    order = desc.arg_order
    if order is None:
        return list(range(len(node.args)))
    if callable(order):
        return order(node)
    return order
