"""
Low-level kernel benchmark: hand-coded gint (SugarProgram, no torch.compile)
vs triton vs torch eager for four kernels — add3, rmsnorm, ggx_importance,
and 4×4 fp32 bmm.

This is a lower-level companion to bench_compile.py: it compares raw
bytecode-kernel quality against hand-written triton kernels, without
conductor / inductor / CUDA-graph overhead.

Usage:
    python benchmark/bench_lowlevel.py --kernel add3
    python benchmark/bench_lowlevel.py --kernel rmsnorm
    python benchmark/bench_lowlevel.py --kernel ggx_importance
    python benchmark/bench_lowlevel.py --kernel bmm4x4
    python benchmark/bench_lowlevel.py --kernel add3 --clear-triton-cache
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

from gint import bytecode, TensorInterface, cdiv
from gint.host.frontend import (
    make_block_1d, fldg_1d, fstg_1d,
    fpush, fmaimm, dup, fma, fmul, fadd, fsub, fdiv, frdiv, frsub,
    warp_allreduce_fsum, fperm_w, frsqrt, halt, pop,
    fmulimm, fcos, fsin, fsqrt, swap,
    fload_reg, fstore_reg,
)
from tests.test_rmsnorm import generalized_rms_norm as triton_rmsnorm

# ---------------------------------------------------------------------------
# Timing helpers (mirror bench_compile.py)
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
# Priming
# ---------------------------------------------------------------------------

def prime_cuda():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    for _ in range(3):
        z = x + y + x
        del z
    torch.cuda.synchronize()


def prime_backends():
    """Pay gint + triton bootstrap on a smoke function (x * 2.0)."""
    x = torch.randn(64, device='cuda', dtype=torch.float32)
    _ = x * 2.0  # eager

    @triton.jit
    def _smoke_triton(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a * 2.0, mask=mask)

    out = torch.empty_like(x)
    _smoke_triton[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)

    # gint smoke: simple add kernel to pay executor bootstrap
    @bytecode
    def _smoke_gint(a, b, y, REGW: int, WARP: int, N: int, M: int):
        a_ti = make_block_1d(a, N, 1, 1, 0, [N], [M])
        b_ti = make_block_1d(b, N, 1, 1, 0, [N], [M])
        y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])
        fldg_1d(0, a_ti)
        fldg_1d(0, b_ti)
        fadd()
        fstg_1d(0, y_ti)
        halt()

    s = torch.randn(128, device='cuda', dtype=torch.float32)
    t = torch.randn(128, device='cuda', dtype=torch.float32)
    u = torch.empty(128, device='cuda', dtype=torch.float32)
    _smoke_gint(s, t, u, N=128, M=1, grid_dim=1)
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


# ---------------------------------------------------------------------------
# Gint kernels
# ---------------------------------------------------------------------------

@bytecode
def gint_add3_kernel(a, b, c, y, REGW: int, WARP: int, N: int, M: int):
    """a + b + c elementwise. M programs, each processes N elements."""
    a_ti = make_block_1d(a, N, 1, 1, 0, [N], [M])
    b_ti = make_block_1d(b, N, 1, 1, 0, [N], [M])
    c_ti = make_block_1d(c, N, 1, 1, 0, [N], [M])
    y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])

    chunk = REGW * WARP
    for off in range(0, N, chunk):
        fldg_1d(off, a_ti)
        fldg_1d(off, b_ti)
        fadd()
        fldg_1d(off, c_ti)
        fadd()
        fstg_1d(off, y_ti)
    halt()


@bytecode
def gint_rmsnorm_singlepass(x, w, y, REGW: int, WARP: int, N: int, M: int):
    """Fused rmsnorm: compute rms-stat and normalize in one pass.

    Caches x in a virtual register so x is loaded from global memory only
    once.  Requires N <= REGW * WARP (single chunk); the caller must ensure
    the shape satisfies this constraint.  One program per row (M rows).
    """
    x_ti = make_block_1d(x, N, 1, 1, 0, [N], [M])
    w_ti = make_block_1d(w, N, 1, 1, 0, [0], [M])
    y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])

    chunk = REGW * WARP  # 128

    fpush(0.0)                       # running sum = 0
    for c in range(0, N, chunk):
        fldg_1d(c, x_ti)             # load x
        dup()                        # x, x
        dup()                        # x, x, x
        fstore_reg(0)                # x, x; reg0 = x (saved for later)
        fma()                        # x^2 + sum
    warp_allreduce_fsum()
    dup(); fperm_w(2, 3, 0, 1); fadd()
    dup(); fperm_w(1, 0, 3, 2); fadd()
    fmaimm(1.0 / N, 1e-5)
    frsqrt()                         # rstd

    for c in range(0, N, chunk):
        dup()                        # rstd, rstd
        fload_reg(0)                 # rstd, rstd, x
        fmul()                       # rstd, x * rstd
        fldg_1d(c, w_ti)             # rstd, x*rstd, w
        fmul()                       # rstd, x*rstd*w
        fstg_1d(c, y_ti)             # rstd
    pop()
    halt()


@bytecode
def gint_ggx_importance_kernel(x, y, hx, hy, hz,
                                REGW: int, WARP: int, N: int, M: int,
                                roughness: float):
    """GGX-D importance-sampled half-vector: (x,y) uniform → (hx,hy,hz).

    Uses the direct sin_theta formula (sin_theta = sqrt(y * a^2 / denom))
    to avoid the precision loss in 1 - cos_theta^2 when cos_theta ≈ 1.

    M programs, each processes N elements.  roughness is a Python float
    folded into bytecode immediates at trace time.
    """
    x_ti = make_block_1d(x, N, 1, 1, 0, [N], [M])
    y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])
    hx_ti = make_block_1d(hx, N, 1, 1, 0, [N], [M])
    hy_ti = make_block_1d(hy, N, 1, 1, 0, [N], [M])
    hz_ti = make_block_1d(hz, N, 1, 1, 0, [N], [M])

    a_sq = roughness * roughness
    a2 = a_sq * a_sq
    a2_m1 = a2 - 1.0

    chunk = REGW * WARP
    for off in range(0, N, chunk):
        # phi = tau * x, cos(phi), sin(phi)
        fldg_1d(off, x_ti)
        fmulimm(2.0 * math.pi)       # phi
        dup()
        fcos()                       # phi, cos(phi)
        fstore_reg(0)                # phi; reg0 = cos(phi)
        fsin()                       # sin(phi)
        fstore_reg(1)                # ; reg1 = sin(phi)

        # denom = 1 + a2_m1 * y  (shared for both cos_theta and sin_theta)
        fldg_1d(off, y_ti)           # y
        dup()                        # y, y
        fmaimm(a2_m1, 1.0)           # y, denom
        dup()                        # y, denom, denom
        fstore_reg(3)                # y, denom; reg3 = denom

        # cos_theta = sqrt((1 - y) / denom)
        swap()                       # denom, y
        fpush(1.0)                   # denom, y, 1.0
        swap()                       # denom, 1.0, y
        frsub()                      # denom, 1-y
        fdiv()                       # (1-y)/denom
        fsqrt()                      # cos_theta
        fstore_reg(2)                # ; reg2 = cos_theta

        # sin_theta = sqrt(y * a^4 / denom)  (direct, avoids 1-cos^2 precision loss)
        fldg_1d(off, y_ti)           # y
        fmulimm(a2)                  # y * a^4
        fload_reg(3)                 # y*a^4, denom
        frdiv()                      # y*a^4 / denom
        fsqrt()                      # sin_theta

        # hy = cos_theta
        dup()                        # sin_theta, sin_theta
        fload_reg(2)                 # sin_theta, sin_theta, cos_theta
        fstg_1d(off, hy_ti)          # sin_theta, sin_theta

        # hx = cos(phi) * sin_theta
        fload_reg(0)                 # sin_theta, sin_theta, cos(phi)
        fmul()                       # sin_theta, hx
        fstg_1d(off, hx_ti)          # sin_theta

        # hz = sin(phi) * sin_theta
        fload_reg(1)                 # sin_theta, sin(phi)
        fmul()                       # hz
        fstg_1d(off, hz_ti)          #
    halt()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _add3_triton(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b + c, mask=mask)




@triton.jit
def _ggx_importance_triton_kernel(x_ptr, y_ptr, hx_ptr, hy_ptr, hz_ptr,
                                   n, ROUGHNESS: tl.constexpr,
                                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xu = tl.load(x_ptr + offs, mask=mask)
    yu = tl.load(y_ptr + offs, mask=mask)
    a = ROUGHNESS * ROUGHNESS
    phi = (2.0 * math.pi) * xu
    cos_theta = tl.sqrt((1.0 - yu) / (1.0 + (a * a - 1.0) * yu))
    sin_theta = tl.sqrt(1.0 - cos_theta * cos_theta)
    tl.store(hx_ptr + offs, tl.cos(phi) * sin_theta, mask=mask)
    tl.store(hz_ptr + offs, tl.sin(phi) * sin_theta, mask=mask)
    tl.store(hy_ptr + offs, cos_theta, mask=mask)


@triton.jit
def _bmm4x4_triton_kernel(a_ptr, b_ptr, c_ptr,
                           B, BLOCK_B: tl.constexpr):
    """Batched 4x4 matmul: c = a @ b. Inputs are (B, 16) row-major.

    One program per batch, loads A and B as (4,4) into registers and
    unrolls the multiply-accumulate.  BLOCK_B batches per program.
    """
    pid = tl.program_id(0)
    batch_start = pid * BLOCK_B
    offs = batch_start + tl.arange(0, BLOCK_B)
    mask = offs < B

    # Helper: load a single element across BLOCK_B batches
    # (inlined — triton doesn't support nested functions)
    a00 = tl.load(a_ptr + offs * 16 + 0, mask=mask, other=0.0)
    a01 = tl.load(a_ptr + offs * 16 + 1, mask=mask, other=0.0)
    a02 = tl.load(a_ptr + offs * 16 + 2, mask=mask, other=0.0)
    a03 = tl.load(a_ptr + offs * 16 + 3, mask=mask, other=0.0)
    a10 = tl.load(a_ptr + offs * 16 + 4, mask=mask, other=0.0)
    a11 = tl.load(a_ptr + offs * 16 + 5, mask=mask, other=0.0)
    a12 = tl.load(a_ptr + offs * 16 + 6, mask=mask, other=0.0)
    a13 = tl.load(a_ptr + offs * 16 + 7, mask=mask, other=0.0)
    a20 = tl.load(a_ptr + offs * 16 + 8, mask=mask, other=0.0)
    a21 = tl.load(a_ptr + offs * 16 + 9, mask=mask, other=0.0)
    a22 = tl.load(a_ptr + offs * 16 + 10, mask=mask, other=0.0)
    a23 = tl.load(a_ptr + offs * 16 + 11, mask=mask, other=0.0)
    a30 = tl.load(a_ptr + offs * 16 + 12, mask=mask, other=0.0)
    a31 = tl.load(a_ptr + offs * 16 + 13, mask=mask, other=0.0)
    a32 = tl.load(a_ptr + offs * 16 + 14, mask=mask, other=0.0)
    a33 = tl.load(a_ptr + offs * 16 + 15, mask=mask, other=0.0)

    # Helper: load a single element across BLOCK_B batches
    # (inlined — triton doesn't support nested functions)
    a00 = tl.load(a_ptr + offs * 16 + 0, mask=mask, other=0.0)
    a01 = tl.load(a_ptr + offs * 16 + 1, mask=mask, other=0.0)
    a02 = tl.load(a_ptr + offs * 16 + 2, mask=mask, other=0.0)
    a03 = tl.load(a_ptr + offs * 16 + 3, mask=mask, other=0.0)
    a10 = tl.load(a_ptr + offs * 16 + 4, mask=mask, other=0.0)
    a11 = tl.load(a_ptr + offs * 16 + 5, mask=mask, other=0.0)
    a12 = tl.load(a_ptr + offs * 16 + 6, mask=mask, other=0.0)
    a13 = tl.load(a_ptr + offs * 16 + 7, mask=mask, other=0.0)
    a20 = tl.load(a_ptr + offs * 16 + 8, mask=mask, other=0.0)
    a21 = tl.load(a_ptr + offs * 16 + 9, mask=mask, other=0.0)
    a22 = tl.load(a_ptr + offs * 16 + 10, mask=mask, other=0.0)
    a23 = tl.load(a_ptr + offs * 16 + 11, mask=mask, other=0.0)
    a30 = tl.load(a_ptr + offs * 16 + 12, mask=mask, other=0.0)
    a31 = tl.load(a_ptr + offs * 16 + 13, mask=mask, other=0.0)
    a32 = tl.load(a_ptr + offs * 16 + 14, mask=mask, other=0.0)
    a33 = tl.load(a_ptr + offs * 16 + 15, mask=mask, other=0.0)

    b00 = tl.load(b_ptr + offs * 16 + 0, mask=mask, other=0.0)
    b01 = tl.load(b_ptr + offs * 16 + 1, mask=mask, other=0.0)
    b02 = tl.load(b_ptr + offs * 16 + 2, mask=mask, other=0.0)
    b03 = tl.load(b_ptr + offs * 16 + 3, mask=mask, other=0.0)
    b10 = tl.load(b_ptr + offs * 16 + 4, mask=mask, other=0.0)
    b11 = tl.load(b_ptr + offs * 16 + 5, mask=mask, other=0.0)
    b12 = tl.load(b_ptr + offs * 16 + 6, mask=mask, other=0.0)
    b13 = tl.load(b_ptr + offs * 16 + 7, mask=mask, other=0.0)
    b20 = tl.load(b_ptr + offs * 16 + 8, mask=mask, other=0.0)
    b21 = tl.load(b_ptr + offs * 16 + 9, mask=mask, other=0.0)
    b22 = tl.load(b_ptr + offs * 16 + 10, mask=mask, other=0.0)
    b23 = tl.load(b_ptr + offs * 16 + 11, mask=mask, other=0.0)
    b30 = tl.load(b_ptr + offs * 16 + 12, mask=mask, other=0.0)
    b31 = tl.load(b_ptr + offs * 16 + 13, mask=mask, other=0.0)
    b32 = tl.load(b_ptr + offs * 16 + 14, mask=mask, other=0.0)
    b33 = tl.load(b_ptr + offs * 16 + 15, mask=mask, other=0.0)

    # C = A @ B  (fully unrolled)
    r0 = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30
    r1 = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31
    r2 = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32
    r3 = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33

    r4 = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30
    r5 = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31
    r6 = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32
    r7 = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33

    r8  = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30
    r9  = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
    r10 = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32
    r11 = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33

    r12 = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30
    r13 = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31
    r14 = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32
    r15 = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33

    # Store C as (B, 16) row-major — one store per output element
    base = offs * 16
    tl.store(c_ptr + base + 0,  r0,  mask=mask)
    tl.store(c_ptr + base + 1,  r1,  mask=mask)
    tl.store(c_ptr + base + 2,  r2,  mask=mask)
    tl.store(c_ptr + base + 3,  r3,  mask=mask)
    tl.store(c_ptr + base + 4,  r4,  mask=mask)
    tl.store(c_ptr + base + 5,  r5,  mask=mask)
    tl.store(c_ptr + base + 6,  r6,  mask=mask)
    tl.store(c_ptr + base + 7,  r7,  mask=mask)
    tl.store(c_ptr + base + 8,  r8,  mask=mask)
    tl.store(c_ptr + base + 9,  r9,  mask=mask)
    tl.store(c_ptr + base + 10, r10, mask=mask)
    tl.store(c_ptr + base + 11, r11, mask=mask)
    tl.store(c_ptr + base + 12, r12, mask=mask)
    tl.store(c_ptr + base + 13, r13, mask=mask)
    tl.store(c_ptr + base + 14, r14, mask=mask)
    tl.store(c_ptr + base + 15, r15, mask=mask)


# ---------------------------------------------------------------------------
# Case: add3  (a + b + c)
# ---------------------------------------------------------------------------

def add3_triton_call(a, b, c):
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 1024
    _add3_triton[(triton.cdiv(n, BLOCK),)](a, b, c, out, n, BLOCK=BLOCK)
    return out


def build_add3(n=1 << 24):
    torch.manual_seed(0)
    a = torch.randn(n, device='cuda', dtype=torch.float32)
    b = torch.randn(n, device='cuda', dtype=torch.float32)
    c = torch.randn(n, device='cuda', dtype=torch.float32)
    y_gint = torch.empty(n, device='cuda', dtype=torch.float32)
    ref = (a + b + c).clone()

    chunk = 128  # REGW * WARP
    M = cdiv(n, chunk)
    N = chunk

    impls = {}
    impls['eager'] = lambda _a=a, _b=b, _c=c: _a + _b + _c
    impls['triton'] = lambda: add3_triton_call(a, b, c)

    def gint_call():
        gint_add3_kernel(a, b, c, y_gint, N=N, M=M, grid_dim=M)
        return y_gint
    impls['gint'] = gint_call

    return impls, ref, f"shape=({n},)", 1e-5, gint_call


# ---------------------------------------------------------------------------
# Case: rmsnorm  (3000×12×64, single-pass gint)
# ---------------------------------------------------------------------------

def build_rmsnorm(B=3000, T=12, H=64):
    """Shape (B, T, H); normalize over dim=-1 (H)."""
    torch.manual_seed(42)
    x_3d = torch.randn(B, T, H, device='cuda', dtype=torch.float32) * 1.5 + 0.5
    w = torch.randn(H, device='cuda', dtype=torch.float32)
    ref = torch.nn.functional.rms_norm(x_3d, (H,), w, eps=1e-5).clone()

    # Flatten to 2D for gint; triton_rmsnorm handles 2D with dim=[1].
    M = B * T
    N = H
    x = x_3d.reshape(M, N)
    y_gint = torch.empty(M, N, device='cuda', dtype=torch.float32)

    impls = {}

    def eager_call():
        return torch.nn.functional.rms_norm(x_3d, (H,), w, eps=1e-5)
    impls['eager'] = eager_call

    impls['triton'] = lambda: triton_rmsnorm(x, x, [1], w, 1e-5).reshape(B, T, H)

    def gint_call():
        gint_rmsnorm_singlepass(x, w, y_gint, N=N, M=M, grid_dim=M)
        return y_gint.reshape(B, T, H)
    impls['gint'] = gint_call

    return impls, ref, f"shape=({B}, {T}, {H})", 1e-4, gint_call


# ---------------------------------------------------------------------------
# Case: ggx_importance
# ---------------------------------------------------------------------------

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
    y = torch.rand(N, device='cuda', dtype=torch.float32)
    roughness = 0.4

    hx_g = torch.empty(N, device='cuda', dtype=torch.float32)
    hy_g = torch.empty(N, device='cuda', dtype=torch.float32)
    hz_g = torch.empty(N, device='cuda', dtype=torch.float32)

    hx_ref, hy_ref, hz_ref = ggx_importance_eager(x, y, roughness)
    ref = torch.stack([hx_ref, hy_ref, hz_ref])

    def stack_call(fn):
        def go():
            hx, hy, hz = fn()
            return torch.stack([hx, hy, hz])
        return go

    chunk = 128  # REGW * WARP
    M = cdiv(N, chunk)
    NN = chunk

    impls = {}
    impls['eager'] = stack_call(lambda: ggx_importance_eager(x, y, roughness))
    impls['triton'] = stack_call(
        lambda: ggx_importance_triton_call(x, y, roughness))

    def gint_call():
        gint_ggx_importance_kernel(
            x, y, hx_g, hy_g, hz_g,
            N=NN, M=M, roughness=roughness, grid_dim=M,
        )
        return hx_g, hy_g, hz_g
    impls['gint'] = stack_call(lambda: gint_call())

    return impls, ref, f"shape=(N={N},)", 1e-2, lambda: gint_call()


# ---------------------------------------------------------------------------
# Case: 4×4 fp32 bmm
# ---------------------------------------------------------------------------

def bmm4x4_eager(a, b):
    return torch.bmm(a.view(-1, 4, 4), b.view(-1, 4, 4)).view(-1, 16)


def bmm4x4_triton_call(a, b):
    """a, b: (B, 16) row-major."""
    B = a.shape[0]
    c = torch.empty(B, 16, device=a.device, dtype=a.dtype)
    BLOCK_B = 128
    _bmm4x4_triton_kernel[(triton.cdiv(B, BLOCK_B),)](
        a, b, c, B, BLOCK_B=BLOCK_B,
    )
    return c


def build_bmm4x4(B=65536):
    torch.manual_seed(42)
    a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
    b = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
    c_gint = torch.empty(B, 4, 4, device='cuda', dtype=torch.float32)
    ref = bmm4x4_eager(a, b).clone()

    from gint.host.matrix import bmm4x4_kernel

    a_flat = a.view(B, 16)
    b_flat = b.view(B, 16)
    c_flat = c_gint.view(B, 16)

    impls = {}
    impls['eager'] = lambda: bmm4x4_eager(a, b)
    impls['triton'] = lambda: bmm4x4_triton_call(a_flat, b_flat)

    def gint_call():
        bmm4x4_kernel(a_flat, b_flat, c_flat, grid_dim=cdiv(B, 32))
        return c_flat
    impls['gint'] = gint_call

    return impls, ref, f"shape=(B={B}, 4, 4)", 1e-4, gint_call


# ---------------------------------------------------------------------------
# Registry + main
# ---------------------------------------------------------------------------

KERNELS = {
    'add3': build_add3,
    'rmsnorm': build_rmsnorm,
    'ggx_importance': build_ggx_importance,
    'bmm4x4': build_bmm4x4,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel', choices=list(KERNELS), default='add3')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--startup-calls', type=int, default=5)
    p.add_argument('--clear-triton-cache', action='store_true')
    args = p.parse_args()

    if args.clear_triton_cache:
        clear_triton_cache()

    prime_cuda()
    prime_backends()

    impls, ref, shape_str, atol, _verify_fn = KERNELS[args.kernel]()

    print(f"\n{args.kernel} low-level benchmark  ({shape_str}, dtype=fp32, "
          f"device={torch.cuda.get_device_name()})")
    print(f"warmup={args.warmup}, iters={args.iters}\n")

    K = args.startup_calls
    header = f"{'impl':<10}" + "".join(f"{'call'+str(i+1):>10}" for i in range(K))
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
        print(f"{name:<10}{early_str}{startup:>12.3f}{runtime:>10.4f}{kernel:>10.4f}")

    print("\nstartup = sum(first K calls) - K * wall  (ms)")
    print("wall    = end-to-end median per-call time (sync around each call)")
    print("kernel  = GPU-only kernel time per call from torch.profiler\n")


if __name__ == '__main__':
    main()