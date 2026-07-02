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

import cuda.bindings.driver as cuda_drv
from cuda.bindings import nvrtc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gint import bytecode, TensorInterface, cdiv
from gint.kernel.interpreter.main import REG_WIDTH
from gint.host.frontend import (
    make_block_1d, make_block_2d,
    fldg_1d, fstg_1d, fldg_2dt, fstg_2dt,
    fldg_1d_f16, fstg_1d_f16, fldg_2dt_f16, fstg_2dt_f16,
    fldg_1d_bf16, fstg_1d_bf16, fldg_2dt_bf16, fstg_2dt_bf16,
    fpush, fmaimm, dup, fma, fmul, fadd, fsub, fdiv, frdiv, frsub,
    warp_allreduce_fsum, fperm_w, frsqrt, halt, pop,
    fmulimm, fcos, fsin, fsqrt, swap,
    fload_reg, fstore_reg,
)
from gint.host.cuda.driver import (
    current_context, check_cuda_error, launch_kernel,
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
# NVRTC: compile CUDA C++ source at runtime to a CUfunction
# ---------------------------------------------------------------------------

def _device_arch_flag() -> bytes:
    err, dev = cuda_drv.cuCtxGetDevice()
    check_cuda_error(err)
    Attr = cuda_drv.CUdevice_attribute
    err, major = cuda_drv.cuDeviceGetAttribute(
        Attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    check_cuda_error(err)
    err, minor = cuda_drv.cuDeviceGetAttribute(
        Attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    check_cuda_error(err)
    return f"-arch=sm_{major}{minor}".encode()


# NVRTC needs an explicit -I to find <cuda_fp16.h> / <cuda_bf16.h>; locate the
# headers shipped with the cuda-runtime wheel (or fall back to the system CUDA
# install).
def _cuda_include_dir() -> str:
    candidates = []
    try:
        import nvidia.cuda_runtime  # type: ignore
        candidates.append(os.path.join(
            os.path.dirname(nvidia.cuda_runtime.__file__), "include"))
    except ImportError:
        pass
    for c in ["/usr/local/cuda/include",
              "/usr/local/cuda-12.1/targets/x86_64-linux/include",
              "/usr/local/cuda-12.4/targets/x86_64-linux/include",
              "/usr/local/cuda-12.8/targets/x86_64-linux/include"]:
        candidates.append(c)
    for c in candidates:
        if os.path.exists(os.path.join(c, "cuda_fp16.h")):
            return c
    raise RuntimeError("Couldn't find cuda_fp16.h; checked: " + repr(candidates))


# Aggressive opts: --use_fast_math (FTZ + low-precision div/sqrt), --restrict
# (treat pointer args as __restrict__), --extra-device-vectorization,
# ptxas allow-expensive-opts = larger search for unrolling/scheduling.
NVRTC_OPTS = [
    b"--use_fast_math",
    b"-default-device",
    b"--restrict",
    b"--extra-device-vectorization",
    # ptxas-options: each flag is a separate --ptxas-options= entry (NVRTC
    # forwards each value verbatim to ptxas).  Default ptxas opt is already
    # -O3; explicitly enable --allow-expensive-optimizations to let it spend
    # more time on scheduling/unroll.
    b"--ptxas-options=--allow-expensive-optimizations=true",
    f"-I{_cuda_include_dir()}".encode(),
]


def _nvrtc_check(res, prog=None, label="nvrtc"):
    err = res[0] if isinstance(res, tuple) else res
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_str = ""
        if prog is not None:
            _, sz = nvrtc.nvrtcGetProgramLogSize(prog)
            buf = bytearray(sz)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            log_str = buf.decode(errors='ignore').strip()
        raise RuntimeError(f"{label} failed ({err}):\n{log_str}")


def nvrtc_compile_module(source: str, file_label: str, fn_names: list[str]):
    """Compile one CUDA C++ source; return dict[fn_name -> CUfunction]."""
    opts = [_device_arch_flag()] + NVRTC_OPTS
    err, prog = nvrtc.nvrtcCreateProgram(
        source.encode(), (file_label + ".cu").encode(), 0, [], [])
    _nvrtc_check(err, label="nvrtcCreateProgram")
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _nvrtc_check(res, prog=prog, label=f"nvrtc compile {file_label}")
    err, sz = nvrtc.nvrtcGetCUBINSize(prog)
    _nvrtc_check(err, prog=prog, label="nvrtcGetCUBINSize")
    cubin = bytearray(sz)
    res = nvrtc.nvrtcGetCUBIN(prog, cubin)
    _nvrtc_check(res, prog=prog, label="nvrtcGetCUBIN")
    nvrtc.nvrtcDestroyProgram(prog)

    ctx = current_context()
    err, mod = cuda_drv.cuModuleLoadData(bytes(cubin))
    check_cuda_error(err)
    ctx.deferred(lambda: check_cuda_error(cuda_drv.cuModuleUnload(mod)))
    out = {}
    for fn in fn_names:
        err, func = cuda_drv.cuModuleGetFunction(mod, fn.encode())
        check_cuda_error(err)
        out[fn] = func
    return out


class LazyCudaModule:
    """Compiles a CUDA C++ source on first .get(name); the compile cost lands
    in the benchmark's first-call timing (the 'startup' column).  All entry
    points live in one module so the compile cost is paid only once even when
    the same source defines several extern-C wrappers (e.g. one per dtype)."""
    def __init__(self, source: str, file_label: str, fn_names: list[str]):
        self.source = source
        self.file_label = file_label
        self.fn_names = fn_names
        self.funcs = None
    def get(self, fn_name: str):
        if self.funcs is None:
            self.funcs = nvrtc_compile_module(
                self.source, self.file_label, self.fn_names)
        return self.funcs[fn_name]


def _dptr(t: torch.Tensor) -> "cuda_drv.CUdeviceptr":
    return cuda_drv.CUdeviceptr(t.data_ptr())


# ---------------------------------------------------------------------------
# CUDA C++ kernel sources (compiled lazily by NVRTC)
# ---------------------------------------------------------------------------

# A small dtype prelude — included by every templated kernel below — defines
# generic to_f / from_f helpers so kernel bodies can stay dtype-agnostic.
DTYPE_PRELUDE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T> __device__ __forceinline__ float  to_f  (T v);
template<> __device__ __forceinline__ float  to_f<float>      (float v)         { return v; }
template<> __device__ __forceinline__ float  to_f<__half>     (__half v)        { return __half2float(v); }
template<> __device__ __forceinline__ float  to_f<__nv_bfloat16>(__nv_bfloat16 v){ return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T      from_f(float v);
template<> __device__ __forceinline__ float  from_f<float>      (float v)       { return v; }
template<> __device__ __forceinline__ __half from_f<__half>     (float v)       { return __float2half_rn(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v){ return __float2bfloat16_rn(v); }
"""

ADD3_CUDA_SRC = DTYPE_PRELUDE + r"""
template<typename T>
__device__ __forceinline__ void add3_impl(
    const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c,
    T* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = from_f<T>(to_f(a[i]) + to_f(b[i]) + to_f(c[i]));
}
extern "C" __global__ void add3_kernel_f32(const float* a, const float* b, const float* c, float* out, int n)
{ add3_impl<float>(a, b, c, out, n); }
extern "C" __global__ void add3_kernel_f16(const __half* a, const __half* b, const __half* c, __half* out, int n)
{ add3_impl<__half>(a, b, c, out, n); }
extern "C" __global__ void add3_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c, __nv_bfloat16* out, int n)
{ add3_impl<__nv_bfloat16>(a, b, c, out, n); }
"""

# RMSNorm over inner dim N (=64).  One warp (32 threads) per row, each thread
# owns N/32 elements; sum-of-squares is reduced via __shfl_xor_sync.
RMSNORM_CUDA_SRC = DTYPE_PRELUDE + r"""
template<typename T>
__device__ __forceinline__ void rmsnorm_impl(
    const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y,
    int M, int N, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;
    const T* xr = x + row * N;
    T* yr = y + row * N;

    const int per_thread = 2;  // N=64, warp=32
    float xs[per_thread];
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < per_thread; ++j) {
        float v = to_f(xr[tid + j * 32]);
        xs[j] = v;
        s += v * v;
    }
    s += __shfl_xor_sync(0xffffffff, s, 16);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);

    float scale = rsqrtf(s / (float)N + eps);
    #pragma unroll
    for (int j = 0; j < per_thread; ++j) {
        yr[tid + j * 32] = from_f<T>(xs[j] * scale * to_f(w[tid + j * 32]));
    }
}
extern "C" __global__ void rmsnorm_kernel_f32(const float* x, const float* w, float* y, int M, int N, float eps)
{ rmsnorm_impl<float>(x, w, y, M, N, eps); }
extern "C" __global__ void rmsnorm_kernel_f16(const __half* x, const __half* w, __half* y, int M, int N, float eps)
{ rmsnorm_impl<__half>(x, w, y, M, N, eps); }
extern "C" __global__ void rmsnorm_kernel_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, int M, int N, float eps)
{ rmsnorm_impl<__nv_bfloat16>(x, w, y, M, N, eps); }
"""

GGX_CUDA_SRC = DTYPE_PRELUDE + r"""
template<typename T>
__device__ __forceinline__ void ggx_impl(
    const T* __restrict__ xu, const T* __restrict__ yu,
    T* __restrict__ hx, T* __restrict__ hy, T* __restrict__ hz,
    int n, float roughness)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = to_f(xu[i]), y = to_f(yu[i]);
    float a = roughness * roughness;
    float phi = 6.28318530717958647692f * x;
    float cos_theta = sqrtf((1.f - y) / (1.f + (a * a - 1.f) * y));
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    float cp, sp;
    __sincosf(phi, &sp, &cp);
    hx[i] = from_f<T>(cp * sin_theta);
    hz[i] = from_f<T>(sp * sin_theta);
    hy[i] = from_f<T>(cos_theta);
}
extern "C" __global__ void ggx_kernel_f32(const float* xu, const float* yu, float* hx, float* hy, float* hz, int n, float roughness)
{ ggx_impl<float>(xu, yu, hx, hy, hz, n, roughness); }
extern "C" __global__ void ggx_kernel_f16(const __half* xu, const __half* yu, __half* hx, __half* hy, __half* hz, int n, float roughness)
{ ggx_impl<__half>(xu, yu, hx, hy, hz, n, roughness); }
extern "C" __global__ void ggx_kernel_bf16(const __nv_bfloat16* xu, const __nv_bfloat16* yu, __nv_bfloat16* hx, __nv_bfloat16* hy, __nv_bfloat16* hz, int n, float roughness)
{ ggx_impl<__nv_bfloat16>(xu, yu, hx, hy, hz, n, roughness); }
"""

# One thread per 4x4 batch; loads A and B into registers, fully-unrolled mul.
BMM4X4_CUDA_SRC = DTYPE_PRELUDE + r"""
template<typename T>
__device__ __forceinline__ void bmm4x4_impl(
    const T* __restrict__ a, const T* __restrict__ b,
    T* __restrict__ c, int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const T* ai = a + i * 16;
    const T* bi = b + i * 16;
    T* ci = c + i * 16;
    float A[16], BB[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) { A[k] = to_f(ai[k]); BB[k] = to_f(bi[k]); }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) s += A[row*4+k] * BB[k*4+col];
            ci[row*4+col] = from_f<T>(s);
        }
    }
}
extern "C" __global__ void bmm4x4_kernel_f32(const float* a, const float* b, float* c, int B)
{ bmm4x4_impl<float>(a, b, c, B); }
"""


# ---------------------------------------------------------------------------
# Gint kernels
# ---------------------------------------------------------------------------

def _make_gint_add3(fldg, fstg):
    @bytecode
    def kernel(a, b, c, y, REGW: int, WARP: int, N: int, M: int):
        a_ti = make_block_1d(a, N, 1, 1, 0, [N], [M])
        b_ti = make_block_1d(b, N, 1, 1, 0, [N], [M])
        c_ti = make_block_1d(c, N, 1, 1, 0, [N], [M])
        y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])
        chunk = REGW * WARP
        for off in range(0, N, chunk):
            fldg(off, a_ti)
            fldg(off, b_ti)
            fadd()
            fldg(off, c_ti)
            fadd()
            fstg(off, y_ti)
        halt()
    return kernel


GINT_ADD3 = {
    'fp32': _make_gint_add3(fldg_1d,      fstg_1d),
    'fp16': _make_gint_add3(fldg_1d_f16,  fstg_1d_f16),
    'bf16': _make_gint_add3(fldg_1d_bf16, fstg_1d_bf16),
}


def _make_gint_rmsnorm(fldg2, fstg2):
    @bytecode
    def kernel(x, w, y, REGW: int, WARP: int, N: int, M: int):
        chunk_t = cdiv(M, REGW)
        x_blk = make_block_2d(x, [N, M], [1, N], [1, chunk_t], [0, REGW])
        w_blk = make_block_2d(w, [N, M], [1, 0], [1, chunk_t], [0, REGW])
        y_blk = make_block_2d(y, [N, M], [1, N], [1, chunk_t], [0, REGW])

        fldg2(0, x_blk); dup(); fmul()
        fldg2(32, x_blk); dup(); fma()
        warp_allreduce_fsum()
        fmaimm(1.0 / N, 1e-5)
        frsqrt()

        dup(); fldg2(0, x_blk); fmul(); fldg2(0, w_blk); fmul(); fstg2(0, y_blk)
        fldg2(32, x_blk); fmul(); fldg2(32, w_blk); fmul(); fstg2(32, y_blk)
        halt()
    return kernel


GINT_RMSNORM = {
    'fp32': _make_gint_rmsnorm(fldg_2dt,      fstg_2dt),
    'fp16': _make_gint_rmsnorm(fldg_2dt_f16,  fstg_2dt_f16),
    'bf16': _make_gint_rmsnorm(fldg_2dt_bf16, fstg_2dt_bf16),
}


def _make_gint_ggx(fldg, fstg):
    @bytecode
    def kernel(x, y, hx, hy, hz,
               REGW: int, WARP: int, N: int, M: int,
               roughness: float):
        x_ti  = make_block_1d(x,  N, 1, 1, 0, [N], [M])
        y_ti  = make_block_1d(y,  N, 1, 1, 0, [N], [M])
        hx_ti = make_block_1d(hx, N, 1, 1, 0, [N], [M])
        hy_ti = make_block_1d(hy, N, 1, 1, 0, [N], [M])
        hz_ti = make_block_1d(hz, N, 1, 1, 0, [N], [M])

        a_sq = roughness * roughness
        a2 = a_sq * a_sq
        a2_m1 = a2 - 1.0

        chunk = REGW * WARP
        for off in range(0, N, chunk):
            fldg(off, x_ti)
            fmulimm(2.0 * math.pi)
            dup(); fcos(); fstore_reg(0)
            fsin(); fstore_reg(1)

            fldg(off, y_ti)
            dup(); fmaimm(a2_m1, 1.0)
            dup(); fstore_reg(3)

            swap(); fpush(1.0); swap(); frsub(); fdiv(); fsqrt(); fstore_reg(2)

            fldg(off, y_ti)
            fmulimm(a2)
            fload_reg(3); frdiv(); fsqrt()

            dup(); fload_reg(2); fstg(off, hy_ti)
            fload_reg(0); fmul(); fstg(off, hx_ti)
            fload_reg(1); fmul(); fstg(off, hz_ti)
        halt()
    return kernel


GINT_GGX = {
    'fp32': _make_gint_ggx(fldg_1d,      fstg_1d),
    'fp16': _make_gint_ggx(fldg_1d_f16,  fstg_1d_f16),
    'bf16': _make_gint_ggx(fldg_1d_bf16, fstg_1d_bf16),
}


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
    xu = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    yu = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    a = ROUGHNESS * ROUGHNESS
    phi = (2.0 * math.pi) * xu
    cos_theta = tl.sqrt((1.0 - yu) / (1.0 + (a * a - 1.0) * yu))
    sin_theta = tl.sqrt(1.0 - cos_theta * cos_theta)
    out_ty = hx_ptr.dtype.element_ty
    tl.store(hx_ptr + offs, (tl.cos(phi) * sin_theta).to(out_ty), mask=mask)
    tl.store(hz_ptr + offs, (tl.sin(phi) * sin_theta).to(out_ty), mask=mask)
    tl.store(hy_ptr + offs, cos_theta.to(out_ty), mask=mask)


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

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# Per-kernel atol per dtype.  Loosened for fp16/bf16 to absorb half-precision
# rounding (eager itself rounds, so the cross-impl diff floor is non-trivial).
ATOL = {
    'add3':           {'fp32': 1e-5, 'fp16': 1e-2, 'bf16': 1e-1},
    'rmsnorm':        {'fp32': 1e-4, 'fp16': 5e-3, 'bf16': 1e-1},
    'ggx_importance': {'fp32': 1e-2, 'fp16': 5e-2, 'bf16': 1e-1},
    'bmm4x4':         {'fp32': 1e-4},
}


def add3_triton_call(a, b, c):
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 1024
    _add3_triton[(triton.cdiv(n, BLOCK),)](a, b, c, out, n, BLOCK=BLOCK)
    return out


def build_add3(dtype='fp32', n=1 << 24):
    tdt = DTYPE_MAP[dtype]
    torch.manual_seed(0)
    a = torch.randn(n, device='cuda', dtype=tdt)
    b = torch.randn(n, device='cuda', dtype=tdt)
    c = torch.randn(n, device='cuda', dtype=tdt)
    y_gint = torch.empty(n, device='cuda', dtype=tdt)
    ref = (a + b + c).clone()

    chunk = 128  # REGW * WARP
    M = cdiv(n, chunk)
    N = chunk
    gint_kernel = GINT_ADD3[dtype]

    impls = {}
    impls['eager'] = lambda _a=a, _b=b, _c=c: _a + _b + _c
    impls['triton'] = lambda: add3_triton_call(a, b, c)

    def gint_call():
        gint_kernel(a, b, c, y_gint, N=N, M=M, grid_dim=M)
        return y_gint
    impls['gint'] = gint_call

    y_cuda = torch.empty(n, device='cuda', dtype=tdt)
    fn_name = f"add3_kernel_{dtype.replace('fp', 'f')}"
    add3_mod = LazyCudaModule(
        ADD3_CUDA_SRC, "add3_kernel",
        ["add3_kernel_f32", "add3_kernel_f16", "add3_kernel_bf16"])
    BLOCK = 1024
    grid = cdiv(n, BLOCK)
    def cuda_call():
        launch_kernel(
            add3_mod.get(fn_name), _dptr(a), _dptr(b), _dptr(c), _dptr(y_cuda), n,
            grid_dim=grid, block_dim=BLOCK,
        )
        return y_cuda
    impls['cuda'] = cuda_call

    return impls, ref, f"shape=({n},)", ATOL['add3'][dtype], gint_call


# ---------------------------------------------------------------------------
# Case: rmsnorm  (3000×12×64, single-pass gint)
# ---------------------------------------------------------------------------

def build_rmsnorm(dtype='fp32', B=3000, T=12, H=64):
    """Shape (B, T, H); normalize over dim=-1 (H)."""
    tdt = DTYPE_MAP[dtype]
    torch.manual_seed(42)
    x_3d = (torch.randn(B, T, H, device='cuda', dtype=torch.float32) * 1.5 + 0.5).to(tdt)
    w = torch.randn(H, device='cuda', dtype=tdt)
    ref = torch.nn.functional.rms_norm(x_3d, (H,), w, eps=1e-5).clone()

    M = B * T
    N = H
    x = x_3d.reshape(M, N)
    y_gint = torch.empty(M, N, device='cuda', dtype=tdt)
    gint_kernel = GINT_RMSNORM[dtype]

    impls = {}

    def eager_call():
        return torch.nn.functional.rms_norm(x_3d, (H,), w, eps=1e-5)
    impls['eager'] = eager_call

    impls['triton'] = lambda: triton_rmsnorm(x, x, [1], w, 1e-5).reshape(B, T, H)

    def gint_call():
        gint_kernel(x, w, y_gint, N=N, M=M, grid_dim=cdiv(M, REG_WIDTH))
        return y_gint.reshape(B, T, H)
    impls['gint'] = gint_call

    y_cuda = torch.empty(M, N, device='cuda', dtype=tdt)
    fn_name = f"rmsnorm_kernel_{dtype.replace('fp', 'f')}"
    rmsnorm_mod = LazyCudaModule(
        RMSNORM_CUDA_SRC, "rmsnorm_kernel",
        ["rmsnorm_kernel_f32", "rmsnorm_kernel_f16", "rmsnorm_kernel_bf16"])
    def cuda_call():
        launch_kernel(
            rmsnorm_mod.get(fn_name),
            _dptr(x), _dptr(w), _dptr(y_cuda), M, N, 1e-5,
            grid_dim=M, block_dim=32,
        )
        return y_cuda.reshape(B, T, H)
    impls['cuda'] = cuda_call

    return impls, ref, f"shape=({B}, {T}, {H})", ATOL['rmsnorm'][dtype], gint_call


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


def build_ggx_importance(dtype='fp32', N=1 << 22):
    tdt = DTYPE_MAP[dtype]
    torch.manual_seed(11)
    x = torch.rand(N, device='cuda', dtype=tdt)
    y = torch.rand(N, device='cuda', dtype=tdt)
    roughness = 0.4

    hx_g = torch.empty(N, device='cuda', dtype=tdt)
    hy_g = torch.empty(N, device='cuda', dtype=tdt)
    hz_g = torch.empty(N, device='cuda', dtype=tdt)

    hx_ref, hy_ref, hz_ref = ggx_importance_eager(x, y, roughness)
    ref = torch.stack([hx_ref, hy_ref, hz_ref])

    def stack_call(fn):
        def go():
            hx, hy, hz = fn()
            return torch.stack([hx, hy, hz])
        return go

    chunk = 128
    M = cdiv(N, chunk)
    NN = chunk
    gint_kernel = GINT_GGX[dtype]

    impls = {}
    impls['eager'] = stack_call(lambda: ggx_importance_eager(x, y, roughness))
    impls['triton'] = stack_call(
        lambda: ggx_importance_triton_call(x, y, roughness))

    def gint_call():
        gint_kernel(x, y, hx_g, hy_g, hz_g,
                    N=NN, M=M, roughness=roughness, grid_dim=M)
        return hx_g, hy_g, hz_g
    impls['gint'] = stack_call(lambda: gint_call())

    hx_c = torch.empty(N, device='cuda', dtype=tdt)
    hy_c = torch.empty(N, device='cuda', dtype=tdt)
    hz_c = torch.empty(N, device='cuda', dtype=tdt)
    fn_name = f"ggx_kernel_{dtype.replace('fp', 'f')}"
    ggx_mod = LazyCudaModule(
        GGX_CUDA_SRC, "ggx_kernel",
        ["ggx_kernel_f32", "ggx_kernel_f16", "ggx_kernel_bf16"])
    BLOCK = 1024
    grid = cdiv(N, BLOCK)
    def cuda_call():
        launch_kernel(
            ggx_mod.get(fn_name),
            _dptr(x), _dptr(y), _dptr(hx_c), _dptr(hy_c), _dptr(hz_c),
            N, float(roughness),
            grid_dim=grid, block_dim=BLOCK,
        )
        return hx_c, hy_c, hz_c
    impls['cuda'] = stack_call(lambda: cuda_call())

    return impls, ref, f"shape=(N={N},)", ATOL['ggx_importance'][dtype], lambda: gint_call()


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


def build_bmm4x4(dtype='fp32', B=65536):
    if dtype != 'fp32':
        raise SystemExit(
            f"bmm4x4 only supports fp32 in this benchmark (got {dtype}); "
            "torch.bmm + gint sugar are fp32-only and the small-N path "
            "doesn't have half-precision instantiations.")
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

    c_cuda = torch.empty(B, 16, device='cuda', dtype=torch.float32)
    bmm_mod = LazyCudaModule(
        BMM4X4_CUDA_SRC, "bmm4x4_kernel", ["bmm4x4_kernel_f32"])
    BLOCK_B = 128
    grid = cdiv(B, BLOCK_B)
    def cuda_call():
        launch_kernel(
            bmm_mod.get("bmm4x4_kernel_f32"),
            _dptr(a_flat), _dptr(b_flat), _dptr(c_cuda), B,
            grid_dim=grid, block_dim=BLOCK_B,
        )
        return c_cuda
    impls['cuda'] = cuda_call

    return impls, ref, f"shape=(B={B}, 4, 4)", ATOL['bmm4x4'][dtype], gint_call


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
    p.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--startup-calls', type=int, default=5)
    p.add_argument('--clear-triton-cache', action='store_true')
    args = p.parse_args()

    if args.clear_triton_cache:
        clear_triton_cache()

    prime_cuda()
    prime_backends()

    impls, ref, shape_str, atol, _verify_fn = KERNELS[args.kernel](dtype=args.dtype)

    print(f"\n{args.kernel} low-level benchmark  ({shape_str}, dtype={args.dtype}, "
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