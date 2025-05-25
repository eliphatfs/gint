"""
The first non-trivial kernel in gint - fused RMSNorm forward!
"""
import torch
import unittest
from gint import ProgramTensorInfo, TensorInterface, bytecode, cdiv
from gint.host.frontend import *


import torch
import triton
import triton.language as tl
from typing import List


@triton.jit
def _generalized_rms_norm_fwd_fused(
    x_stat,  # pointer to the input to pow - mean - add - rsqrt
    x_normed,  # pointer to the input to normalize
    Y,  # pointer to the output
    W,  # pointer to the weights
    x_stride_r: tl.constexpr, x_stride_c: tl.constexpr,
    y_stride_r: tl.constexpr, y_stride_c: tl.constexpr,
    w_stride_c: tl.constexpr,
    M: tl.constexpr,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    THREAD_ILP_SIZE: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0) * THREAD_ILP_SIZE
    Y_ptr = tl.make_block_ptr(Y, (M, N), (y_stride_r, y_stride_c), (row, 0), (THREAD_ILP_SIZE, BLOCK_SIZE), (0, 1))
    W_ptr = tl.make_block_ptr(W, (N,), (w_stride_c,), (0,), (BLOCK_SIZE,), (0,))
    x_stat_ptr = tl.make_block_ptr(x_stat, (M, N), (x_stride_r, x_stride_c), (row, 0), (THREAD_ILP_SIZE, BLOCK_SIZE), (0, 1))
    x_normed_ptr = tl.make_block_ptr(x_normed, (M, N), (x_stride_r, x_stride_c), (row, 0), (THREAD_ILP_SIZE, BLOCK_SIZE), (0, 1))
    # Compute mean
    rms = tl.zeros([THREAD_ILP_SIZE], dtype=tl.float32)
    _rms = tl.zeros([THREAD_ILP_SIZE, BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        a = tl.load(x_stat_ptr, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
        _rms += a * a
        x_stat_ptr = tl.advance(x_stat_ptr, (0, BLOCK_SIZE))
    rms = tl.sum(_rms, axis=1, keep_dims=True) / N
    rstd = tl.rsqrt(rms + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        w = tl.load(W_ptr, boundary_check=(0,), padding_option='zero').to(tl.float32)
        x = tl.load(x_normed_ptr, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y_ptr, y.to(Y.type.element_ty),  boundary_check=(0,), cache_modifier=".cg")
        x_normed_ptr = tl.advance(x_normed_ptr, (0, BLOCK_SIZE))
        W_ptr = tl.advance(W_ptr, (BLOCK_SIZE,))
        Y_ptr = tl.advance(Y_ptr, (0, BLOCK_SIZE))


def ensure_semi_contiguous(x: torch.Tensor, column_dim: int, msg_name: str):
    # a tensor is semi contiguous if all elements can be traversed according to strides of a row dimension
    strides = list(x.stride())
    shape = list(x.shape)
    column_stride = strides.pop(column_dim)
    column_size = shape.pop(column_dim)
    order_strides = sorted(range(len(strides)), key=strides.__getitem__)
    expected_stride = strides[order_strides[0]]
    row_size = 1
    for arg in order_strides:
        stride = strides[arg]
        size = shape[arg]
        if stride != expected_stride:
            raise ValueError("RMSNorm tensor not semi-contiguous:", msg_name)
        expected_stride *= size
        row_size *= size
    return strides[order_strides[0]], column_stride, row_size, column_size


@torch.library.custom_op("infernity::generalized_rms_norm", mutates_args=(), device_types="cuda")
def generalized_rms_norm(x_stat: torch.Tensor, x_normed: torch.Tensor, dim: List[int], weight: torch.Tensor, eps: float) -> torch.Tensor:
    assert x_stat.stride() == x_normed.stride(), "x_stat and x_normed need to have the same strides"
    assert len(dim) == 1, "only 1 dimension can be normalized by rms norm, got %s" % dim
    x_stride_r, x_stride_c, M, N = ensure_semi_contiguous(x_stat, dim[0], "x_stat")
    y = torch.empty_like(x_normed)
    y_stride_r, y_stride_c, _, _ = ensure_semi_contiguous(y, dim[0], "y")
    assert list(weight.shape) == [N], "weight shape mismatch"
    # reshape input data into 2D tensor
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // 4
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This rms norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    THREAD_ILP_SIZE = 8 // num_warps
    BLOCKS = (M + THREAD_ILP_SIZE - 1) // THREAD_ILP_SIZE
    _generalized_rms_norm_fwd_fused[(BLOCKS, )](  #
        x_stat, x_normed, y, weight,
        x_stride_r, x_stride_c,
        y_stride_r, y_stride_c,
        weight.stride(0),
        M, N, eps,  #
        BLOCK_SIZE=BLOCK_SIZE,
        THREAD_ILP_SIZE=THREAD_ILP_SIZE,
        num_warps=num_warps,
        num_ctas=1
    )
    return y


@bytecode
def rmsnorm(x: TensorInterface, y: TensorInterface, w: TensorInterface, ILP: int, WARP: int):
    # x: B, NH, T, H
    assert x.shape == y.shape
    B, NH, T, H = x.shape
    assert (H,) == tuple(w.shape)
    
    ldtinfos()
    # compute sum of head
    for c in range(0, H, WARP):
        ldg_f1_bf16(c, x)  # r, x[c:c+32], 0, 0
        movf(2, 1)  # r, x[c:c+32], x[c:c+32], 0
        fma_f0_f1_f2()
    warp_allreduce_sum_f0()
    immf(1, 1 / H)
    mul_f0_f1()
    frsqrt_f0()
    immf(1, 1e-5)
    add_f0_f1()
    movf(3, 0)
    for c in range(0, H, WARP):
        movf(0, 3)  # rstd, x[c: c+32], 0, rstd
        ldg_f1_bf16(c, x)  # rstd, x[c:c+32], 0, rstd
        mul_f0_f1()
        ldg_f1_bf16(c, w)  # r * x, w, 0, rstd
        mul_f0_f1()
        stg_f0_bf16(c, y)
    halt()
    return (
        [ProgramTensorInfo(2, a.strides[-1], H, list(a.strides[:3]), [B, NH, T], [0, 0, 0]) for a in (x, y)]
        + [ProgramTensorInfo(2, w.strides[-1], H, [0, 0, 0], [B, NH, T], [0, 0, 0])]
    )


class TestRMSNorm(unittest.TestCase):
    
    def test_fused_rmsnorm_fwd(self):
        for b in [1, 4]:
            for t in [12, 800, 3000]:
                for nh in [1, 6, 12, 24]:
                    for h in [16, 32, 64, 80, 128]:
                        torch.manual_seed(42)
                        x = torch.randn(b, t, nh, h, device='cuda', dtype=torch.bfloat16).transpose(1, 2) * 1.5 + 0.5
                        y = torch.empty_like(x)
                        w = torch.rand(h, device='cuda', dtype=torch.bfloat16)
                        rmsnorm(x, y, w, grid_dim=b * nh * cdiv(t, ILP))
                        y_ref = torch.nn.functional.rms_norm(x, (h,), w, eps=1e-5)
                        torch.testing.assert_close(y, y_ref)

    def test_profile_rmsnorm_fwd(self):
        torch.manual_seed(42)
        b, t, nh, h = 1, 3000, 12, 64
        x = (torch.randn(b, t, nh, h, dtype=torch.bfloat16) * 1.5 + 0.5).cuda().transpose(1, 2)
        y = torch.empty_like(x)
        w = torch.rand(h, device='cuda', dtype=torch.bfloat16)
        rmsnorm(x, y, w, grid_dim=b * nh * cdiv(t, ILP))
        generalized_rms_norm(x, x, [3], w, 1e-5)
        torch.nn.functional.rms_norm(x, (h,), w, eps=1e-5)
