"""
Compile-and-runtime benchmark for four kernels — add3 (a+b+c), rmsnorm,
geometry_smith (BRDF Smith geometry term from diffrp), and
ggx_importance (GGX-D importance-sampled half-vector head from diffrp) —
across:
  - eager torch
  - torch.jit.script
  - triton (inline kernel for add3, generalized_rms_norm from tests for rmsnorm)
  - torch.compile(backend='inductor')                            (no cuda graphs)
  - torch.compile(backend='inductor', mode='reduce-overhead')    (cuda graphs)
  - torch.compile(backend='gint')                                (cuda graphs)
  - torch.compile(backend='gint', options={"cuda_graphs": False})  (no cuda graphs)

Per-backend priming with a different small function (x * 2.0) pays the
one-time backend-bootstrap cost up front, so the timed first call to the
real kernel reflects per-function compile cost only.

PyTorch keeps two on-disk triton caches: ~/.triton/cache/ (used by
standalone triton calls) and /tmp/torchinductor_<user>/triton/ (used when
@triton.jit kernels are invoked from PyTorch — including the rmsnorm
kernel from tests). The same path also stores inductor's codegened
artifacts. Pass --clear-triton-cache to wipe BOTH of these caches before
the run for honest cold-compile numbers; this also gives cold inductor
numbers as a side effect.

Usage:
    python benchmark/bench_compile.py --kernel add3
    python benchmark/bench_compile.py --kernel rmsnorm
    python benchmark/bench_compile.py --kernel geometry_smith
    python benchmark/bench_compile.py --kernel ggx_importance
    python benchmark/bench_compile.py --kernel add3 --clear-triton-cache
"""

import argparse
import math
import os
import pathlib
import shutil
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gint.conductor  # noqa: F401
from tests.test_rmsnorm import generalized_rms_norm as triton_rmsnorm
from gint import bytecode, TensorInterface
from gint.host.frontend import (
    make_block_1d, fldg_1d, fstg_1d, fpush, fmaimm, dup, fma, fmul, fadd,
    warp_allreduce_fsum, fperm_w, frsqrt, halt, pop,
)


@bytecode
def gint_rmsnorm_manual_fp32(
    x: TensorInterface, w: TensorInterface, y: TensorInterface,
    REGW: int, WARP: int, N: int, M: int,
):
    """Hand-rolled fused fp32 rms_norm: one warp per row, one kernel.
    Mirrors the structure of the bf16 4D kernel in tests/test_rmsnorm.py
    but for fp32 contiguous (M, N). Sets the upper bound on what the
    interpreter can do for this shape — anything slower in the conductor
    is fusion / scheduling overhead, not interpreter cost."""
    x_ti = make_block_1d(x, N, 1, 1, 0, [N], [M])
    w_ti = make_block_1d(w, N, 1, 1, 0, [0], [M])
    y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])

    chunk = REGW * WARP  # 128
    fpush(0.0)
    for c in range(0, N, chunk):
        fldg_1d(c, x_ti)
        dup()
        fma()
    warp_allreduce_fsum()
    dup(); fperm_w(2, 3, 0, 1); fadd()
    dup(); fperm_w(1, 0, 3, 2); fadd()
    fmaimm(1.0 / N, 1e-5)
    frsqrt()

    for c in range(0, N, chunk):
        dup()
        fldg_1d(c, x_ti)
        fmul()
        fldg_1d(c, w_ti)
        fmul()
        fstg_1d(c, y_ti)
    pop()
    halt()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_call_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def median_ms(fn, warmup, iters) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1000.0


def kernel_time_ms(fn, warmup=10, iters=50) -> float:
    """Pure GPU kernel time per call, via torch.profiler. Strips Python /
    dispatch overhead — what ncu would show. Use this for kernel-quality
    comparison; wall-clock is for end-to-end-call comparison."""
    import torch.profiler as tp
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with tp.profile(activities=[tp.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    return total_us / iters / 1000.0


# ---------------------------------------------------------------------------
# CUDA + per-backend priming
# ---------------------------------------------------------------------------

def prime_cuda():
    """Burn off first-time CUDA context init / allocator warmup."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    for _ in range(3):
        z = x + y + x
        del z
    torch.cuda.synchronize()


def _smoke_eager(x):
    return x * 2.0


@triton.jit
def _smoke_triton_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a * 2.0, mask=mask)


def _smoke_triton(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    _smoke_triton_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out


def prime_backends():
    """Run x*2.0 through every backend so one-time bootstrap is paid up front.
    Deliberately a different function from any of the timed kernels."""
    x = torch.randn(64, device='cuda', dtype=torch.float32)

    _ = _smoke_eager(x)

    @torch.jit.script
    def smoke_jit(t):
        return t * 2.0
    for _ in range(3):  # cross TorchScript profiler threshold
        _ = smoke_jit(x)

    _ = _smoke_triton(x)

    _ = torch.compile(_smoke_eager, backend='inductor')(x)
    _ = torch.compile(_smoke_eager, backend='inductor', mode='reduce-overhead')(x)
    _ = torch.compile(_smoke_eager, backend='gint', options={"clone_outputs": False})(x)
    _ = torch.compile(_smoke_eager, backend='gint', options={"cuda_graphs": False})(x)
    torch.cuda.synchronize()


def clear_triton_cache():
    """Wipe both triton caches: ~/.triton/cache/ (standalone triton) and
    /tmp/torchinductor_<user>/ (PyTorch's wrapper, which actually serves
    @triton.jit kernels invoked from torch). Without clearing the second
    one, @triton.jit kernels can be served from a prior run's cache."""
    user = os.environ.get('USER') or os.environ.get('LOGNAME') or 'root'
    targets = [
        pathlib.Path.home() / ".triton" / "cache",
        pathlib.Path(f"/tmp/torchinductor_{user}"),
    ]
    for cache in targets:
        if cache.exists():
            shutil.rmtree(cache)
            print(f"Cleared {cache}")
        else:
            print(f"No cache at {cache}")


def verify(name, ref, got, atol):
    if not torch.allclose(ref, got, atol=atol, rtol=1e-3):
        diff = (ref - got).abs().max().item()
        print(f"  [warn] {name}: max abs diff = {diff:.3e}")


# ---------------------------------------------------------------------------
# Kernel: a + b + c
# ---------------------------------------------------------------------------

@triton.jit
def _add3_triton_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b + c, mask=mask)


def add3_triton_call(a, b, c):
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 1024
    _add3_triton_kernel[(triton.cdiv(n, BLOCK),)](a, b, c, out, n, BLOCK=BLOCK)
    return out


def add3_eager(a, b, c):
    return a + b + c


def build_add3(n):
    torch.manual_seed(0)
    a = torch.randn(n, device='cuda', dtype=torch.float32)
    b = torch.randn(n, device='cuda', dtype=torch.float32)
    c = torch.randn(n, device='cuda', dtype=torch.float32)
    ref = (a + b + c).clone()

    impls = {}
    impls['eager'] = lambda: add3_eager(a, b, c)

    @torch.jit.script
    def jit_fn(x, y, z):
        return x + y + z
    impls['torch.jit'] = lambda: jit_fn(a, b, c)

    impls['triton'] = lambda: add3_triton_call(a, b, c)

    impls['inductor'] = (lambda f=torch.compile(add3_eager, backend='inductor'):
                         f(a, b, c))
    impls['inductor-rg'] = (lambda f=torch.compile(add3_eager, backend='inductor',
                                                   mode='reduce-overhead'):
                            f(a, b, c))
    impls['gint'] = (lambda f=torch.compile(add3_eager, backend='gint', options={"clone_outputs": False}):
                     f(a, b, c))
    impls['gint-nocg'] = (lambda f=torch.compile(add3_eager,
                                                 backend='gint', options={"cuda_graphs": False}):
                          f(a, b, c))
    return impls, ref, f"shape=({n},)", 1e-5


# ---------------------------------------------------------------------------
# Kernel: rms_norm
# ---------------------------------------------------------------------------

def rms_norm_eager(x, w, eps=1e-5):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps=eps)


def rms_norm_manual(x, w, eps: float = 1e-5):
    """Hand-decomposed rms_norm using pow/mean/rsqrt/mul, exposing the op
    chain to the TorchScript NNC fuser (which sees aten::rms_norm as a
    single opaque builtin and cannot fuse through it)."""
    rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * rstd * w


def build_rmsnorm(M=4096, N=2048):
    """Default shape (4096, 2048) puts each tensor at 32 MB — well into the
    bandwidth-bound regime where actual kernel quality dominates wall time
    (per-pass cost ~64 µs at 1 TB/s, vs ~30 µs Python-dispatch overhead).
    The original test shape (36000, 64) is dispatch-bound: 2.3 MB tensors
    where the triton custom_op wrapper Python overhead (~30 µs) drowns out
    the actual kernel time (~5 µs)."""
    torch.manual_seed(42)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32) * 1.5 + 0.5
    w = torch.rand(N, device='cuda', dtype=torch.float32)
    ref = rms_norm_eager(x, w).clone()

    impls = {}
    impls['eager'] = lambda: rms_norm_eager(x, w)

    @torch.jit.script
    def jit_fn(t, weight):
        n = t.shape[-1]
        return torch.nn.functional.rms_norm(t, [n], weight, eps=1e-5)
    impls['jit-script'] = lambda: jit_fn(x, w)

    # torch.jit.trace of the eager rms_norm — records whatever ATen op the
    # eager path actually dispatches (in 2.5.0, a single aten::rms_norm).
    traced = torch.jit.trace(rms_norm_eager, (x, w))
    impls['jit-trace'] = lambda: traced(x, w)

    # torch.jit.script of the manually decomposed rms_norm — exposes the
    # pow/mean/rsqrt/mul chain to NNC fusion.
    @torch.jit.script
    def jit_manual(t, weight, eps: float = 1e-5):
        rstd = torch.rsqrt(torch.mean(t * t, dim=-1, keepdim=True) + eps)
        return t * rstd * weight
    impls['jit-manual'] = lambda: jit_manual(x, w)

    # Triton's generalized_rms_norm takes (x_stat, x_normed, dim, weight, eps).
    impls['triton'] = lambda: triton_rmsnorm(x, x, [1], w, 1e-5)

    impls['inductor'] = (lambda f=torch.compile(rms_norm_eager, backend='inductor'):
                         f(x, w))
    impls['inductor-rg'] = (lambda f=torch.compile(rms_norm_eager, backend='inductor',
                                                   mode='reduce-overhead'):
                            f(x, w))
    # gint uses rms_norm_manual: torch.compile decomposes F.rms_norm to
    # pow(x, 2) + mean + add + rsqrt + mul + mul, and pow is not in
    # OP_REGISTRY, so the eager form falls back entirely. The manual form
    # uses mul(x, x), which gint compiles into a real subgraph chain.
    impls['gint'] = (lambda f=torch.compile(rms_norm_manual, backend='gint', options={"clone_outputs": False}):
                     f(x, w))
    impls['gint-nocg'] = (lambda f=torch.compile(rms_norm_manual,
                                                 backend='gint', options={"cuda_graphs": False}):
                          f(x, w))

    # Hand-rolled fused fp32 rmsnorm (single gint kernel) — upper bound on
    # what the interpreter can deliver for this shape. Compares directly
    # against the conductor's 3-subgraph chain.
    y = torch.empty_like(x)
    def gint_manual_call():
        gint_rmsnorm_manual_fp32(x, w, y, N=N, M=M, grid_dim=M)
        return y
    impls['gint-manual'] = gint_manual_call

    return impls, ref, f"shape=({M}, {N})", 1e-4


# ---------------------------------------------------------------------------
# Kernel: geometry_smith (BRDF Smith geometry term, from diffrp)
# ---------------------------------------------------------------------------
# https://github.com/eliphatfs/diffrp/blob/main/diffrp/utils/light_transport.py
#
# Two inner-dim-3 dot products (sum-reductions on a tiny dim), each fed
# through relu and a Schlick-GGX rational, then combined with a final
# multiply. Reduction dim is 3 — the conductor's reduction kernel pads
# OOB lanes with 0.0 so sum tolerates non-multiple-of-128. Exercises the
# pre-reduction (n*v fused into the reduction) and post-reduction
# (relu + schlick_ggx + roughness broadcast) fusion paths in one shot.

def _gs_schlick_ggx(n_dot_x, roughness):
    a = roughness
    k = (a * a) / 2.0
    return n_dot_x / (n_dot_x * (1.0 - k) + k)


def geometry_smith_eager(n, v, L, roughness):
    n_dot_v = torch.relu((n * v).sum(-1, keepdim=True))
    n_dot_L = torch.relu((n * L).sum(-1, keepdim=True))
    ggx2 = _gs_schlick_ggx(n_dot_v, roughness)
    ggx1 = _gs_schlick_ggx(n_dot_L, roughness)
    return ggx1 * ggx2


@triton.jit
def _gs_triton_kernel(n_ptr, v_ptr, L_ptr, r_ptr, out_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK + tl.arange(0, BLOCK)
    mask = base < M
    idx3 = base[:, None] * 3 + tl.arange(0, 4)[None, :]
    sub3 = tl.arange(0, 4)[None, :] < 3
    mask3 = mask[:, None] & sub3
    nv = tl.load(n_ptr + idx3, mask=mask3, other=0.0)
    vv = tl.load(v_ptr + idx3, mask=mask3, other=0.0)
    Lv = tl.load(L_ptr + idx3, mask=mask3, other=0.0)
    r = tl.load(r_ptr + base, mask=mask, other=0.0)

    n_dot_v = tl.maximum(tl.sum(nv * vv, axis=1), 0.0)
    n_dot_L = tl.maximum(tl.sum(nv * Lv, axis=1), 0.0)
    k = (r * r) * 0.5
    om_k = 1.0 - k
    ggx2 = n_dot_v / (n_dot_v * om_k + k)
    ggx1 = n_dot_L / (n_dot_L * om_k + k)
    tl.store(out_ptr + base, ggx1 * ggx2, mask=mask)


def geometry_smith_triton_call(n, v, L, r):
    M = n.shape[0]
    out = torch.empty(M, 1, device=n.device, dtype=n.dtype)
    BLOCK = 256
    _gs_triton_kernel[(triton.cdiv(M, BLOCK),)](
        n, v, L, r, out, M, BLOCK=BLOCK,
    )
    return out


def build_geometry_smith(M=1 << 20):
    """diffrp BRDF Smith geometry term. M=2^20 puts the (M, 3) inputs at
    12 MB each (~44 MB total working set) — bandwidth-bound, so kernel
    quality dominates. Tests reduction fusion + multi-launch dispatch."""
    torch.manual_seed(7)
    n = torch.randn(M, 3, device='cuda', dtype=torch.float32)
    v = torch.randn(M, 3, device='cuda', dtype=torch.float32)
    L = torch.randn(M, 3, device='cuda', dtype=torch.float32)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    L = L / L.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    r = torch.rand(M, 1, device='cuda', dtype=torch.float32) * 0.9 + 0.1
    ref = geometry_smith_eager(n, v, L, r).clone()

    impls = {}
    impls['eager'] = lambda: geometry_smith_eager(n, v, L, r)

    @torch.jit.script
    def jit_fn(nn, vv, ll, rr):
        n_dot_v = torch.relu((nn * vv).sum(-1, keepdim=True))
        n_dot_L = torch.relu((nn * ll).sum(-1, keepdim=True))
        a = rr
        k = (a * a) / 2.0
        om_k = 1.0 - k
        ggx2 = n_dot_v / (n_dot_v * om_k + k)
        ggx1 = n_dot_L / (n_dot_L * om_k + k)
        return ggx1 * ggx2
    impls['torch.jit'] = lambda: jit_fn(n, v, L, r)

    impls['triton'] = lambda: geometry_smith_triton_call(n, v, L, r)

    impls['inductor'] = (lambda f=torch.compile(geometry_smith_eager, backend='inductor'):
                         f(n, v, L, r))
    impls['inductor-rg'] = (lambda f=torch.compile(geometry_smith_eager, backend='inductor',
                                                   mode='reduce-overhead'):
                            f(n, v, L, r))
    impls['gint'] = (lambda f=torch.compile(geometry_smith_eager, backend='gint', options={"clone_outputs": False}):
                     f(n, v, L, r))
    impls['gint-nocg'] = (lambda f=torch.compile(geometry_smith_eager,
                                                 backend='gint', options={"cuda_graphs": False}):
                          f(n, v, L, r))
    return impls, ref, f"shape=(M={M}, 3)", 1e-4


# ---------------------------------------------------------------------------
# Kernel: ggx_importance (GGX-D importance-sampled half-vector, from diffrp)
# ---------------------------------------------------------------------------
# https://github.com/eliphatfs/diffrp/blob/main/diffrp/utils/light_transport.py#L87
#
# Float-roughness branch of GGX importance sampling: per-sample (x, y) →
# half-vector (hx, hy, hz) before the tangent-space rotation. 4
# transcendentals (sqrt × 2, cos, sin) plus a scalar-folded rational —
# every elementwise op uses the Tensor-Scalar variant since `roughness`
# is a Python float. Three independent outputs in one fused subgraph
# (no reduction, no broadcast suffix). Pure pointwise / 1D bandwidth
# bound at large N — what differentiates backends here is per-output
# launch dispatch and how trig-heavy chains schedule.

def ggx_importance_eager(x, y, roughness):
    a = roughness * roughness
    phi = math.tau * x
    cos_theta = torch.sqrt((1.0 - y) / (1.0 + (a * a - 1.0) * y))
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
    hx = torch.cos(phi) * sin_theta
    hz = torch.sin(phi) * sin_theta
    hy = cos_theta
    return hx, hy, hz


@triton.jit
def _ggx_importance_triton_kernel(x_ptr, y_ptr, hx_ptr, hy_ptr, hz_ptr,
                                  n, ROUGHNESS: tl.constexpr,
                                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    a = ROUGHNESS * ROUGHNESS
    phi = (2.0 * math.pi) * x
    cos_theta = tl.sqrt((1.0 - y) / (1.0 + (a * a - 1.0) * y))
    sin_theta = tl.sqrt(1.0 - cos_theta * cos_theta)
    tl.store(hx_ptr + offs, tl.cos(phi) * sin_theta, mask=mask)
    tl.store(hz_ptr + offs, tl.sin(phi) * sin_theta, mask=mask)
    tl.store(hy_ptr + offs, cos_theta, mask=mask)


def ggx_importance_triton_call(x, y, roughness):
    n = x.numel()
    hx = torch.empty_like(x)
    hy = torch.empty_like(x)
    hz = torch.empty_like(x)
    BLOCK = 1024
    _ggx_importance_triton_kernel[(triton.cdiv(n, BLOCK),)](
        x, y, hx, hy, hz, n, ROUGHNESS=roughness, BLOCK=BLOCK,
    )
    return hx, hy, hz


def build_ggx_importance(N=1 << 22):
    """N=2^22 samples ≈ 16 MB per tensor (5 tensors total in/out → 80 MB),
    bandwidth-bound regime. Roughness is a fixed Python float so it folds
    into Tensor-Scalar ops on the gint side and a constexpr on triton."""
    torch.manual_seed(11)
    x = torch.rand(N, device='cuda', dtype=torch.float32)
    y = torch.rand(N, device='cuda', dtype=torch.float32)
    roughness = 0.4

    hx_ref, hy_ref, hz_ref = ggx_importance_eager(x, y, roughness)
    # Stack outputs so verify() can do a single allclose.
    ref = torch.stack([hx_ref, hy_ref, hz_ref])

    def stack_call(fn):
        def go():
            hx, hy, hz = fn()
            return torch.stack([hx, hy, hz])
        return go

    impls = {}
    impls['eager'] = stack_call(lambda: ggx_importance_eager(x, y, roughness))

    @torch.jit.script
    def jit_fn(xx, yy, rr: float):
        a = rr * rr
        phi = (2.0 * math.pi) * xx
        cos_theta = torch.sqrt((1.0 - yy) / (1.0 + (a * a - 1.0) * yy))
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        hx = torch.cos(phi) * sin_theta
        hz = torch.sin(phi) * sin_theta
        hy = cos_theta
        return hx, hy, hz
    impls['torch.jit'] = stack_call(lambda: jit_fn(x, y, roughness))

    impls['triton'] = stack_call(lambda: ggx_importance_triton_call(x, y, roughness))

    impls['inductor'] = stack_call(
        lambda f=torch.compile(ggx_importance_eager, backend='inductor'):
        f(x, y, roughness))
    impls['inductor-rg'] = stack_call(
        lambda f=torch.compile(ggx_importance_eager, backend='inductor',
                               mode='reduce-overhead'):
        f(x, y, roughness))
    impls['gint'] = stack_call(
        lambda f=torch.compile(ggx_importance_eager, backend='gint', options={"clone_outputs": False}):
        f(x, y, roughness))
    impls['gint-nocg'] = stack_call(
        lambda f=torch.compile(ggx_importance_eager,
                               backend='gint', options={"cuda_graphs": False}):
        f(x, y, roughness))
    return impls, ref, f"shape=(N={N},)", 1e-4


KERNELS = {
    'add3': build_add3,
    'rmsnorm': build_rmsnorm,
    'geometry_smith': build_geometry_smith,
    'ggx_importance': build_ggx_importance,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel', choices=list(KERNELS), default='add3')
    p.add_argument('--n', type=int, default=1 << 24,
                   help='vector length for add3 (ignored for rmsnorm)')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--startup-calls', type=int, default=5)
    p.add_argument('--clear-triton-cache', action='store_true',
                   help='Wipe ~/.triton/cache/ before running for cold-compile numbers')
    args = p.parse_args()

    if args.clear_triton_cache:
        clear_triton_cache()

    prime_cuda()
    prime_backends()

    if args.kernel == 'add3':
        impls, ref, shape_str, atol = build_add3(args.n)
    else:
        impls, ref, shape_str, atol = KERNELS[args.kernel]()

    print(f"\n{args.kernel} benchmark  ({shape_str}, dtype=fp32, "
          f"device={torch.cuda.get_device_name()})")
    print(f"warmup={args.warmup}, iters={args.iters}\n")

    K = args.startup_calls
    header = f"{'impl':<14}" + "".join(f"{'call'+str(i+1):>10}" for i in range(K))
    header += f"{'startup':>12}{'wall':>10}{'kernel':>10}"
    print(header)
    print('-' * len(header))

    for name, fn in impls.items():
        early = [time_call_ms(fn) for _ in range(K)]
        verify(name, ref, fn(), atol=atol)
        runtime = median_ms(fn, args.warmup, args.iters)
        kernel = kernel_time_ms(fn)
        startup = sum(early) - K * runtime
        early_str = "".join(f"{t:>10.3f}" for t in early)
        print(f"{name:<14}{early_str}{startup:>12.3f}{runtime:>10.4f}{kernel:>10.4f}")

    print("\nstartup = sum(first K calls) - K * wall  (ms)")
    print("wall    = end-to-end median per-call time (sync around each call)")
    print("kernel  = GPU-only kernel time per call from torch.profiler "
          "(strips Python/dispatch overhead, ~ncu)\n")


if __name__ == '__main__':
    main()
