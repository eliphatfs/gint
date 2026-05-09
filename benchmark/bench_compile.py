"""
Compile-and-runtime benchmark for four kernels — add3 (a+b+c), rmsnorm,
geometry_smith (BRDF Smith geometry term from diffrp), and
ggx_importance (GGX-D importance-sampled half-vector head from diffrp) —
across:
  - eager torch
  - torch.jit.script
  - torch.compile(backend='inductor')                            (no cuda graphs)
  - torch.compile(backend='inductor', mode='reduce-overhead')    (cuda graphs)
  - torch.compile(backend='gint')                                (cuda graphs)
  - torch.compile(backend='gint', options={"cuda_graphs": False})  (no cuda graphs)

Reports forward then backward (fwd+bwd) startup / wall / kernel time.
The first differentiable argument receives .grad from loss.sum().backward().
gint is inference-only so backward is skipped for gint backends.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gint.conductor  # noqa: F401


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
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    for _ in range(3):
        z = x + y + x
        del z
    torch.cuda.synchronize()


def _smoke_eager(x):
    return x * 2.0


def prime_backends():
    x = torch.randn(64, device='cuda', dtype=torch.float32)
    _ = _smoke_eager(x)

    @torch.jit.script
    def smoke_jit(t):
        return t * 2.0
    for _ in range(3):
        _ = smoke_jit(x)

    _ = torch.compile(_smoke_eager, backend='inductor')(x)
    _ = torch.compile(_smoke_eager, backend='inductor', mode='reduce-overhead')(x)
    _ = torch.compile(_smoke_eager, backend='gint', options={"clone_outputs": False})(x)
    _ = torch.compile(_smoke_eager, backend='gint', options={"cuda_graphs": False})(x)
    torch.cuda.synchronize()


def clear_triton_cache():
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


def _run_bench(impls, ref, atol, warmup, iters, startup_calls, label):
    K = startup_calls
    header = f"{'impl':<14}" + "".join(f"{'call'+str(i+1):>10}" for i in range(K))
    header += f"{'startup':>12}{'wall':>10}{'kernel':>10}"
    print(f"\n--- {label} ---")
    print(header)
    print('-' * len(header))

    for name, fn in impls.items():
        early = [time_call_ms(fn) for _ in range(K)]
        out = fn()
        if isinstance(out, tuple):
            verify(name, ref, torch.stack(out), atol=atol)
        else:
            verify(name, ref, out, atol=atol)
        runtime = median_ms(fn, warmup, iters)
        kernel = kernel_time_ms(fn)
        startup = sum(early) - K * runtime
        early_str = "".join(f"{t:>10.3f}" for t in early)
        print(f"{name:<14}{early_str}{startup:>12.3f}{runtime:>10.4f}{kernel:>10.4f}")


# ---------------------------------------------------------------------------
# Backward helpers
# ---------------------------------------------------------------------------

def _make_bwd(fw_call, grad_tensor, is_tuple=False):
    """Create a fwd+bwd callable. grad_tensor.grad is zeroed before each call.
    Returns the forward output for verification."""
    if is_tuple:
        def go():
            grad_tensor.grad = None
            outs = fw_call()
            loss = sum(o.sum() for o in outs)
            loss.backward()
            return outs
        return go
    else:
        def go():
            grad_tensor.grad = None
            out = fw_call()
            out.sum().backward()
            return out
        return go


def _bwd_compile(callable_fn, fn, backend, mode, options, args, grad_tensor, is_tuple):
    """Create a backward callable that compiles *fn* on first invocation.
    Uses a fresh compiled instance so forward/backward compilations are
    independent of the forward-benchmark compiled instances."""
    compiled = None
    def fw_call():
        nonlocal compiled
        if compiled is None:
            if mode is not None:
                compiled = torch.compile(fn, backend=backend, mode=mode)
            elif options is not None:
                compiled = torch.compile(fn, backend=backend, options=options)
            else:
                compiled = torch.compile(fn, backend=backend)
        return compiled(*args)
    return callable_fn(fw_call, grad_tensor, is_tuple)


# ---------------------------------------------------------------------------
# Kernel: a + b + c
# ---------------------------------------------------------------------------

def add3_eager(a, b, c):
    return a + b + c


def build_add3(n):
    torch.manual_seed(0)
    a = torch.randn(n, device='cuda', dtype=torch.float32, requires_grad=True)
    b = torch.randn(n, device='cuda', dtype=torch.float32)
    c = torch.randn(n, device='cuda', dtype=torch.float32)
    ref = (a + b + c).clone()

    # -- jit (shared between fwd and bwd) --
    @torch.jit.script
    def f_jit(x, y, z):
        return x + y + z

    # -- forward impls --
    impls = {
        'eager': lambda: add3_eager(a, b, c),
        'jit': lambda: f_jit(a, b, c),
        'inductor': (lambda f=torch.compile(add3_eager, backend='inductor'): f(a, b, c)),
        'inductor-rg': (lambda f=torch.compile(add3_eager, backend='inductor', mode='reduce-overhead'): f(a, b, c)),
        'gint': (lambda f=torch.compile(add3_eager, backend='gint', options={"clone_outputs": False}): f(a.detach(), b, c)),
        'gint-nocg': (lambda f=torch.compile(add3_eager, backend='gint', options={"cuda_graphs": False}): f(a.detach(), b, c)),
    }

    # -- backward impls (fresh compile closures) --
    bwd_impls = {}
    bwd_impls['eager'] = _make_bwd(lambda: add3_eager(a, b, c), a)
    bwd_impls['jit'] = _make_bwd(lambda: f_jit(a, b, c), a)
    bwd_impls['inductor'] = _bwd_compile(_make_bwd, add3_eager, 'inductor', None, None, (a, b, c), a, False)
    bwd_impls['inductor-rg'] = _bwd_compile(_make_bwd, add3_eager, 'inductor', 'reduce-overhead', None, (a, b, c), a, False)
    bwd_impls['gint-nocg'] = _bwd_compile(_make_bwd, add3_eager, 'gint', None, {'cuda_graphs': False, 'clone_outputs': False}, (a, b, c), a, False)

    return impls, bwd_impls, ref, f"shape=({n},)", 1e-5


# ---------------------------------------------------------------------------
# Kernel: rms_norm
# ---------------------------------------------------------------------------

def rms_norm_eager(x, w, eps=1e-5):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps=eps)


def rms_norm_manual(x, w, eps: float = 1e-5):
    rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * rstd * w


def build_rmsnorm(M=4096, N=2048):
    torch.manual_seed(42)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32) * 1.5 + 0.5
    x = x.detach().requires_grad_(True)  # leaf tensor for clean backward
    w = torch.rand(N, device='cuda', dtype=torch.float32)
    ref = rms_norm_eager(x.detach(), w).clone()

    @torch.jit.script
    def f_jit(t, weight, eps: float = 1e-5):
        rstd = torch.rsqrt(torch.mean(t * t, dim=-1, keepdim=True) + eps)
        return t * rstd * weight

    # -- forward impls --
    impls = {
        'eager': lambda: rms_norm_manual(x, w),
        'jit': lambda: f_jit(x, w),
        'inductor': (lambda f=torch.compile(rms_norm_manual, backend='inductor'): f(x, w)),
        'inductor-rg': (lambda f=torch.compile(rms_norm_manual, backend='inductor', mode='reduce-overhead'): f(x, w)),
        'gint': (lambda f=torch.compile(rms_norm_manual, backend='gint', options={"clone_outputs": False}): f(x.detach(), w)),
        'gint-nocg': (lambda f=torch.compile(rms_norm_manual, backend='gint', options={"cuda_graphs": False}): f(x.detach(), w)),
    }

    # -- backward impls --
    bwd_impls = {}
    bwd_impls['eager'] = _make_bwd(lambda: rms_norm_manual(x, w), x)
    bwd_impls['jit'] = _make_bwd(lambda: f_jit(x, w), x)
    bwd_impls['inductor'] = _bwd_compile(_make_bwd, rms_norm_manual, 'inductor', None, None, (x, w), x, False)
    bwd_impls['inductor-rg'] = _bwd_compile(_make_bwd, rms_norm_manual, 'inductor', 'reduce-overhead', None, (x, w), x, False)
    bwd_impls['gint-nocg'] = _bwd_compile(_make_bwd, rms_norm_manual, 'gint', None, {'cuda_graphs': False, 'clone_outputs': False}, (x, w), x, False)

    return impls, bwd_impls, ref, f"shape=({M}, {N})", 1e-4


# ---------------------------------------------------------------------------
# Kernel: geometry_smith (BRDF Smith geometry term, from diffrp)
# ---------------------------------------------------------------------------

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


def build_geometry_smith(M=1 << 20):
    torch.manual_seed(7)
    n = torch.randn(M, 3, device='cuda', dtype=torch.float32)
    v = torch.randn(M, 3, device='cuda', dtype=torch.float32)
    L = torch.randn(M, 3, device='cuda', dtype=torch.float32)
    n_raw = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    L = L / L.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    r = torch.rand(M, 1, device='cuda', dtype=torch.float32) * 0.9 + 0.1
    n = n_raw.detach().requires_grad_(True)  # leaf tensor
    ref = geometry_smith_eager(n, v, L, r).clone()

    @torch.jit.script
    def f_jit(nn, vv, ll, rr):
        n_dot_v = torch.relu((nn * vv).sum(-1, keepdim=True))
        n_dot_L = torch.relu((nn * ll).sum(-1, keepdim=True))
        a = rr
        k = (a * a) / 2.0
        om_k = 1.0 - k
        ggx2 = n_dot_v / (n_dot_v * om_k + k)
        ggx1 = n_dot_L / (n_dot_L * om_k + k)
        return ggx1 * ggx2

    # -- forward impls --
    impls = {
        'eager': lambda: geometry_smith_eager(n, v, L, r),
        'jit': lambda: f_jit(n, v, L, r),
        'inductor': (lambda f=torch.compile(geometry_smith_eager, backend='inductor'): f(n, v, L, r)),
        'inductor-rg': (lambda f=torch.compile(geometry_smith_eager, backend='inductor', mode='reduce-overhead'): f(n, v, L, r)),
        'gint': (lambda f=torch.compile(geometry_smith_eager, backend='gint', options={"clone_outputs": False}): f(n.detach(), v, L, r)),
        'gint-nocg': (lambda f=torch.compile(geometry_smith_eager, backend='gint', options={"cuda_graphs": False}): f(n.detach(), v, L, r)),
    }

    # -- backward impls --
    bwd_impls = {}
    bwd_impls['eager'] = _make_bwd(lambda: geometry_smith_eager(n, v, L, r), n)
    bwd_impls['jit'] = _make_bwd(lambda: f_jit(n, v, L, r), n)
    bwd_impls['inductor'] = _bwd_compile(_make_bwd, geometry_smith_eager, 'inductor', None, None, (n, v, L, r), n, False)
    bwd_impls['inductor-rg'] = _bwd_compile(_make_bwd, geometry_smith_eager, 'inductor', 'reduce-overhead', None, (n, v, L, r), n, False)
    bwd_impls['gint-nocg'] = _bwd_compile(_make_bwd, geometry_smith_eager, 'gint', None, {'cuda_graphs': False, 'clone_outputs': False}, (n, v, L, r), n, False)

    return impls, bwd_impls, ref, f"shape=(M={M}, 3)", 1e-4


# ---------------------------------------------------------------------------
# Kernel: ggx_importance (GGX-D importance-sampled half-vector, from diffrp)
# ---------------------------------------------------------------------------

def ggx_importance_eager(x, y, roughness):
    a = roughness * roughness
    phi = math.tau * x
    cos_theta = torch.sqrt((1.0 - y) / (1.0 + (a * a - 1.0) * y))
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
    hx = torch.cos(phi) * sin_theta
    hz = torch.sin(phi) * sin_theta
    hy = cos_theta
    return hx, hy, hz


def build_ggx_importance(N=1 << 22):
    torch.manual_seed(11)
    x = torch.rand(N, device='cuda', dtype=torch.float32)
    x = x.detach().requires_grad_(True)  # leaf tensor
    y = torch.rand(N, device='cuda', dtype=torch.float32)
    roughness = 0.4

    hx_ref, hy_ref, hz_ref = ggx_importance_eager(x, y, roughness)
    ref = torch.stack([hx_ref, hy_ref, hz_ref])

    def stack_call(fn):
        def go():
            hx, hy, hz = fn()
            return torch.stack([hx, hy, hz])
        return go

    @torch.jit.script
    def f_jit(xx, yy, rr: float):
        a = rr * rr
        phi = (2.0 * math.pi) * xx
        cos_theta = torch.sqrt((1.0 - yy) / (1.0 + (a * a - 1.0) * yy))
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        hx = torch.cos(phi) * sin_theta
        hz = torch.sin(phi) * sin_theta
        hy = cos_theta
        return hx, hy, hz

    # -- forward impls (stack for single-tensor verification) --
    impls = {
        'eager': stack_call(lambda: ggx_importance_eager(x, y, roughness)),
        'jit': stack_call(lambda: f_jit(x, y, roughness)),
        'inductor': stack_call(
            lambda f=torch.compile(ggx_importance_eager, backend='inductor'):
            f(x, y, roughness)),
        'inductor-rg': stack_call(
            lambda f=torch.compile(ggx_importance_eager, backend='inductor', mode='reduce-overhead'):
            f(x, y, roughness)),
        'gint': stack_call(
            lambda f=torch.compile(ggx_importance_eager, backend='gint', options={"clone_outputs": False}):
            f(x.detach(), y, roughness)),
        'gint-nocg': stack_call(
            lambda f=torch.compile(ggx_importance_eager, backend='gint', options={"cuda_graphs": False}):
            f(x.detach(), y, roughness)),
    }

    # -- backward impls (tuple outputs) --
    bwd_impls = {}
    bwd_impls['eager'] = _make_bwd(lambda: ggx_importance_eager(x, y, roughness), x, is_tuple=True)
    bwd_impls['jit'] = _make_bwd(lambda: f_jit(x, y, roughness), x, is_tuple=True)
    bwd_impls['inductor'] = _bwd_compile(_make_bwd, ggx_importance_eager, 'inductor', None, None, (x, y, roughness), x, True)
    bwd_impls['inductor-rg'] = _bwd_compile(_make_bwd, ggx_importance_eager, 'inductor', 'reduce-overhead', None, (x, y, roughness), x, True)
    # gint-nocg omitted: AOT backward for trig-heavy ops needs > 8 global
    # slots after fixing the burial issue, which exceeds max_tensors.

    return impls, bwd_impls, ref, f"shape=(N={N},)", 1e-4


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
                   help='Wipe ~/.triton/cache/ and /tmp/torchinductor_* before running')
    args = p.parse_args()

    if args.clear_triton_cache:
        clear_triton_cache()

    prime_cuda()
    prime_backends()

    if args.kernel == 'add3':
        impls, bwd_impls, ref, shape_str, atol = build_add3(args.n)
    else:
        impls, bwd_impls, ref, shape_str, atol = KERNELS[args.kernel]()

    print(f"\n{args.kernel} benchmark  ({shape_str}, dtype=fp32, "
          f"device={torch.cuda.get_device_name()})")
    print(f"warmup={args.warmup}, iters={args.iters}")

    _run_bench(impls, ref, atol, args.warmup, args.iters, args.startup_calls, "forward")
    _run_bench(bwd_impls, ref, atol, args.warmup, args.iters, args.startup_calls,
               "backward (fwd+bwd, loss.sum().backward())")

    print("\nstartup = sum(first K calls) - K * wall  (ms)")
    print("wall    = end-to-end median per-call time (sync around each call)")
    print("kernel  = GPU-only kernel time per call from torch.profiler "
          "(strips Python/dispatch overhead, ~ncu)")
    print("gint (with cuda graphs) omitted from backward: CUDA graph replay strips autograd metadata\n")


if __name__ == '__main__':
    main()