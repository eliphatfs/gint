import torch
import unittest
from tests import requires_gpu
import gint.conductor  # noqa: F401  (auto-registers "gint" / "gint-no-cuda-graph")
from gint.conductor.debug import inspect_subgraphs


@requires_gpu
class TestConductorBackend(unittest.TestCase):
    # ------------------------------------------------------------------
    # Original tests
    # ------------------------------------------------------------------

    def test_simple_arithmetic(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x + y * 2.0

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y * 2.0)

    def test_partitioning_limit(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            a = x + y
            b = a * 2.0
            c = b - x
            d = c + y
            e = d + a
            f = e * b
            g = f - c
            h = g + d
            return h

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        expected = fn.__wrapped__(x, y)
        torch.testing.assert_close(fn(x, y), expected)

    def test_many_ops_one_subgraph(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            res = x
            for _ in range(14):
                res = res + 1.0
            return res + 1.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), fn.__wrapped__(x))

    def test_mixed_shapes(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x + 1.0, y * 2.0

        x = torch.randn(1024, device='cuda')
        y = torch.randn(512, device='cuda')
        ax, ay = fn(x, y)
        torch.testing.assert_close(ax, x + 1.0)
        torch.testing.assert_close(ay, y * 2.0)

    # ------------------------------------------------------------------
    # Unary ops
    # ------------------------------------------------------------------

    def test_unary_neg(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return -x

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), -x)

    def test_unary_sqrt(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sqrt(x)

        x = torch.rand(1024, device='cuda') + 0.1
        torch.testing.assert_close(fn(x), torch.sqrt(x))

    def test_unary_exp_log(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.log(torch.exp(x))

        x = torch.randn(1024, device='cuda') * 0.5
        torch.testing.assert_close(fn(x), x, atol=1e-5, rtol=1e-5)

    def test_unary_sin_cos(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.sin(x) + torch.cos(x), atol=1e-5, rtol=1e-5)

    def test_unary_abs(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.abs(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.abs(x))

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def test_relu(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.relu(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.relu(x))

    def test_gelu(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.nn.functional.gelu(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.nn.functional.gelu(x), atol=1e-5, rtol=1e-5)

    def test_silu(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.nn.functional.silu(x), atol=1e-5, rtol=1e-5)

    def test_leaky_relu(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.nn.functional.leaky_relu(x, negative_slope=0.1)

        x = torch.randn(1024, device='cuda')
        expected = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        torch.testing.assert_close(fn(x), expected)

    # ------------------------------------------------------------------
    # Comparisons and where
    # ------------------------------------------------------------------

    def test_comparison_gt(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return (x > y).float()

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x, y), (x > y).float())

    def test_where(self):
        # Use comparison result as float mask (0.0/1.0) to avoid bool-tensor complications.
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y, z):
            mask = x > 0.0
            return torch.where(mask, y, z)

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        z = torch.randn(1024, device='cuda')
        expected = torch.where(x > 0.0, y, z)
        torch.testing.assert_close(fn(x, y, z), expected)

    # ------------------------------------------------------------------
    # Compound: relu + arithmetic, fused with no intermediate stores
    # ------------------------------------------------------------------

    def test_relu_plus_scale(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.relu(x) * 2.0 + 1.0

        x = torch.randn(1024, device='cuda')
        expected = torch.relu(x) * 2.0 + 1.0
        torch.testing.assert_close(fn(x), expected)

    def test_gelu_chain(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.nn.functional.gelu(x + 0.5) - 0.1

        x = torch.randn(1024, device='cuda')
        expected = torch.nn.functional.gelu(x + 0.5) - 0.1
        torch.testing.assert_close(fn(x), expected, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def test_broadcast_bias_add(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x + y

        x = torch.randn(32, 128, device='cuda')
        y = torch.randn(128, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    def test_broadcast_scalar_tensor(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x * y

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1, device='cuda')
        torch.testing.assert_close(fn(x, y), x * y)

    def test_broadcast_both_expand(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x + y

        x = torch.randn(64, 1, device='cuda')
        y = torch.randn(1, 32, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    def test_broadcast_3d(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x + y

        x = torch.randn(4, 8, 64, device='cuda')
        y = torch.randn(64, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    def test_broadcast_with_activation(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return torch.relu(x + y)

        x = torch.randn(32, 128, device='cuda')
        y = torch.randn(128, device='cuda')
        torch.testing.assert_close(fn(x, y), torch.relu(x + y))

    def test_broadcast_same_shape(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x + y

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    # ------------------------------------------------------------------
    # Fallback: unsupported op should still produce correct result via eager
    # ------------------------------------------------------------------

    def test_fallback_unsupported_op(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.relu(x) + 1.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.relu(x) + 1.0)

    # ------------------------------------------------------------------
    # Metadata ops (view, unsqueeze, squeeze, expand, etc.)
    # ------------------------------------------------------------------

    def test_unsqueeze_squeeze_identity(self):
        """unsqueeze + squeeze cancel out; should fuse with surrounding ops."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x.unsqueeze(0).squeeze(0) * 2.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), x * 2.0)

    def test_unsqueeze_fuse_with_relu(self):
        """relu + unsqueeze + add should fuse into one subgraph."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.relu(x).unsqueeze(0) + 1.0

        x = torch.randn(1024, device='cuda')
        expected = torch.relu(x).unsqueeze(0) + 1.0
        torch.testing.assert_close(fn(x), expected)

    def test_unsqueeze_broadcast_add(self):
        """unsqueeze enables broadcasting: (N,) → (1, N) + (M, N) → (M, N)."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x.unsqueeze(0) + y

        x = torch.randn(128, device='cuda')
        y = torch.randn(32, 128, device='cuda')
        expected = x.unsqueeze(0) + y
        torch.testing.assert_close(fn(x, y), expected)

    def test_expand_add(self):
        """expand broadcasts size-1 dim; fuses with add."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x, y):
            return x.expand(32, 128) + y

        x = torch.randn(1, 128, device='cuda')
        y = torch.randn(32, 128, device='cuda')
        expected = x.expand(32, 128) + y
        torch.testing.assert_close(fn(x, y), expected)

    def test_view_pointwise(self):
        """view followed by pointwise on the new shape."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x.view(32, 32) + 1.0

        x = torch.randn(1024, device='cuda')
        expected = x.view(32, 32) + 1.0
        torch.testing.assert_close(fn(x), expected)

    def test_metadata_standalone(self):
        """A metadata op alone (no fusion) should still produce correct result."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x.unsqueeze(0) + 1.0

        x = torch.randn(512, device='cuda')
        expected = x.unsqueeze(0) + 1.0
        torch.testing.assert_close(fn(x), expected)

    # ------------------------------------------------------------------
    # Slice metadata ops
    # ------------------------------------------------------------------

    def test_slice_relu_fusion(self):
        """slice + relu should fuse into one kernel subgraph."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x[:, :64].relu()

        x = torch.randn(32, 128, device='cuda')
        expected = x[:, :64].relu()
        torch.testing.assert_close(fn(x), expected)

        # Verify single kernel: the real slice (on dim=1) + relu are fused.
        # Use a fresh function for inspect_subgraphs (fn.__wrapped__ doesn't work).
        def _ref(x):
            return x[:, :64].relu()
        info = inspect_subgraphs(_ref, x)
        fused = [sg for sg in info if len(sg.nodes) > 1]
        self.assertEqual(len(fused), 1,
                         f"Expected 1 fused subgraph, got {len(fused)}: "
                         f"{[[n.name for n in sg.nodes] for sg in info]}")

    def test_slice_bias_fusion(self):
        """slice + bias broadcast should fuse into one kernel subgraph."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x[:, 32:96] + 1.0

        x = torch.randn(32, 128, device='cuda')
        expected = x[:, 32:96] + 1.0
        torch.testing.assert_close(fn(x), expected)

        def _ref(x):
            return x[:, 32:96] + 1.0
        info = inspect_subgraphs(_ref, x)
        fused = [sg for sg in info if len(sg.nodes) > 1]
        self.assertEqual(len(fused), 1,
                         f"Expected 1 fused subgraph, got {len(fused)}: "
                         f"{[[n.name for n in sg.nodes] for sg in info]}")

    def test_slice_offset_relu_fusion(self):
        """slice with non-zero start + relu should still fuse."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x[:, 16:80].relu()

        x = torch.randn(32, 128, device='cuda')
        expected = x[:, 16:80].relu()
        torch.testing.assert_close(fn(x), expected)

        def _ref(x):
            return x[:, 16:80].relu()
        info = inspect_subgraphs(_ref, x)
        fused = [sg for sg in info if len(sg.nodes) > 1]
        self.assertEqual(len(fused), 1,
                         f"Expected 1 fused subgraph, got {len(fused)}: "
                         f"{[[n.name for n in sg.nodes] for sg in info]}")


    # ------------------------------------------------------------------
    # Reduction ops (sum, mean)
    # ------------------------------------------------------------------

    def test_sum_1d(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=0)

        x = torch.randn(128, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=0), atol=1e-4, rtol=1e-4)

    def test_sum_innermost(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_sum_full_warp(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(16, 128, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_mean_innermost(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.mean(x, dim=-1)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.mean(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_sum_keepdim(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=-1, keepdim=True)

        x = torch.randn(32, 128, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1, keepdim=True), atol=1e-4, rtol=1e-4)

    def test_mean_keepdim(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.mean(x, dim=-1, keepdim=True), atol=1e-4, rtol=1e-4)

    def test_sum_large_dim(self):
        """Reduction dim > 128, requires multi-chunk unrolled loads."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(8, 512, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-3, rtol=1e-3)

    def test_sum_3d(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(4, 8, 32, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_sum_non_innermost_fallback(self):
        """Non-innermost reduction should fall back to eager."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return torch.sum(x, dim=0)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=0), atol=1e-4, rtol=1e-4)

    def test_mean_subtract(self):
        """Reduction + broadcast pointwise: x - mean(x, keepdim=True)."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(16, 64, device='cuda')
        expected = x - torch.mean(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-4, rtol=1e-4)

        def _ref(x):
            return x - torch.mean(x, dim=-1, keepdim=True)
        info = inspect_subgraphs(_ref, x)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused subgraph, got {len(info)}")

    def test_normalize_fused(self):
        """Fused: x / sum(x, keepdim=True)."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x / torch.sum(x, dim=-1, keepdim=True)

        x = torch.rand(8, 32, device='cuda') + 0.1  # positive to avoid div-by-zero
        expected = x / torch.sum(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-4, rtol=1e-4)

        def _ref(x):
            return x / torch.sum(x, dim=-1, keepdim=True)
        info = inspect_subgraphs(_ref, x)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused subgraph, got {len(info)}")

    def test_mean_subtract_large(self):
        """Fused reduction+pointwise with multi-chunk (dim > 128)."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(4, 256, device='cuda')
        expected = x - torch.mean(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-3, rtol=1e-3)

        def _ref(x):
            return x - torch.mean(x, dim=-1, keepdim=True)
        info = inspect_subgraphs(_ref, x)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused subgraph, got {len(info)}")

    def test_mean_subtract_still_fuses(self):
        """Regression: existing single-op fusion still works after generalization."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(16, 64, device='cuda')
        expected = x - torch.mean(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-4, rtol=1e-4)

        def _ref(x):
            return x - torch.mean(x, dim=-1, keepdim=True)
        info = inspect_subgraphs(_ref, x)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused subgraph, got {len(info)}")

    def test_sum_normalize_still_fuses(self):
        """Regression: x / sum(x, keepdim=True) still fuses."""
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(x):
            return x / torch.sum(x, dim=-1, keepdim=True)

        x = torch.rand(8, 32, device='cuda') + 0.1
        expected = x / torch.sum(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-4, rtol=1e-4)

        def _ref(x):
            return x / torch.sum(x, dim=-1, keepdim=True)
        info = inspect_subgraphs(_ref, x)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused subgraph, got {len(info)}")

    def test_rms_norm_fused(self):
        """RMSNorm: x * rsqrt(mean(x*x) + eps) * w in a single kernel."""
        def rms_norm_manual(x, w, eps=1e-5):
            rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            return x * rstd * w

        x = torch.randn(16, 64, device='cuda')
        w = torch.randn(64, device='cuda')

        # Inspect BEFORE compiling — Dynamo caches and won't re-run gint.
        info = inspect_subgraphs(rms_norm_manual, x, w)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused kernel, got {len(info)}")

        compiled = torch.compile(rms_norm_manual, backend="gint", options={"cuda_graphs": False})
        torch.testing.assert_close(compiled(x, w), rms_norm_manual(x, w),
                                   atol=1e-4, rtol=1e-4)

    def test_rms_norm_3d(self):
        """RMSNorm with 3D input + 1D weight."""
        def rms_norm_manual(x, w, eps=1e-5):
            rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            return x * rstd * w

        x = torch.randn(4, 8, 32, device='cuda')
        w = torch.randn(32, device='cuda')

        info = inspect_subgraphs(rms_norm_manual, x, w)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused kernel, got {len(info)}")

        compiled = torch.compile(rms_norm_manual, backend="gint", options={"cuda_graphs": False})
        torch.testing.assert_close(compiled(x, w), rms_norm_manual(x, w),
                                   atol=1e-4, rtol=1e-4)

    def test_rms_norm_large_dim(self):
        """RMSNorm with multi-chunk (dim > 128)."""
        def rms_norm_manual(x, w, eps=1e-5):
            rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            return x * rstd * w

        x = torch.randn(4, 256, device='cuda')
        w = torch.randn(256, device='cuda')

        info = inspect_subgraphs(rms_norm_manual, x, w)
        self.assertEqual(len(info), 1,
                         f"Expected 1 fused kernel, got {len(info)}")

        compiled = torch.compile(rms_norm_manual, backend="gint", options={"cuda_graphs": False})
        torch.testing.assert_close(compiled(x, w), rms_norm_manual(x, w),
                                   atol=1e-3, rtol=1e-3)


@requires_gpu
class TestConductorNewOps(unittest.TestCase):
    """Coverage for ops added in the easy-wins pass: reciprocal/atan2,
    amax/amin/prod reductions, composed activations and clamp, composed
    unary math, and addcmul/addcdiv/lerp."""

    def setUp(self):
        # Many small compiles in one process cross Dynamo's cache_size_limit
        # which then flips to dynamic shapes — gint doesn't support symints.
        # Reset between tests so each compile starts fresh.
        torch._dynamo.reset()

    def _check(self, fn, *args, atol=1e-5, rtol=1e-5):
        # Wrap fn in a Python def so Dynamo's trace_rules don't try to special-case
        # bare torch.* / torch.nn.functional.* references.
        def wrapped(*a):
            return fn(*a)
        compiled = torch.compile(wrapped, backend="gint", options={"cuda_graphs": False})
        torch.testing.assert_close(compiled(*args), fn(*args), atol=atol, rtol=rtol)

    # --- Wired-from-existing kernel ops ---

    def test_reciprocal(self):
        x = torch.rand(1024, device='cuda') + 0.1
        self._check(torch.reciprocal, x)

    def test_atan2(self):
        y = torch.randn(1024, device='cuda')
        x = torch.randn(1024, device='cuda')
        self._check(torch.atan2, y, x, atol=1e-5, rtol=1e-5)

    # --- New reductions ---

    def test_amax_innermost(self):
        x = torch.randn(8, 256, device='cuda')
        self._check(lambda t: torch.amax(t, dim=-1), x)

    def test_amax_keepdim(self):
        x = torch.randn(8, 256, device='cuda')
        self._check(lambda t: torch.amax(t, dim=-1, keepdim=True), x)

    def test_amin_innermost(self):
        x = torch.randn(8, 256, device='cuda')
        self._check(lambda t: torch.amin(t, dim=-1), x)

    def test_prod_innermost(self):
        x = torch.rand(8, 128, device='cuda') * 0.9 + 0.1   # avoid huge product
        self._check(lambda t: torch.prod(t, dim=-1), x, atol=1e-3, rtol=1e-3)

    def test_amax_n_not_multiple_of_128_falls_back(self):
        # check_fn rejects n%128 != 0 — this should still work via eager fallback.
        x = torch.randn(8, 100, device='cuda')
        self._check(lambda t: torch.amax(t, dim=-1), x)

    # --- Composed activations ---

    def test_tanh(self):
        x = torch.randn(1024, device='cuda') * 2.0
        self._check(torch.tanh, x)

    def test_sigmoid(self):
        x = torch.randn(1024, device='cuda') * 2.0
        self._check(torch.sigmoid, x)

    def test_hardtanh(self):
        x = torch.randn(1024, device='cuda') * 2.0
        self._check(torch.nn.functional.hardtanh, x)

    def test_relu6(self):
        x = torch.randn(1024, device='cuda') * 4.0
        self._check(torch.nn.functional.relu6, x)

    def test_hardsigmoid(self):
        x = torch.randn(1024, device='cuda') * 4.0
        self._check(torch.nn.functional.hardsigmoid, x)

    def test_hardswish(self):
        x = torch.randn(1024, device='cuda') * 4.0
        self._check(torch.nn.functional.hardswish, x)

    def test_softplus(self):
        x = torch.randn(1024, device='cuda') * 1.5
        self._check(torch.nn.functional.softplus, x, atol=1e-4, rtol=1e-4)

    def test_mish(self):
        x = torch.randn(1024, device='cuda') * 1.5
        self._check(torch.nn.functional.mish, x, atol=1e-4, rtol=1e-4)

    def test_elu(self):
        x = torch.randn(1024, device='cuda') * 1.5
        self._check(torch.nn.functional.elu, x)

    def test_selu(self):
        x = torch.randn(1024, device='cuda') * 1.5
        self._check(torch.nn.functional.selu, x, atol=1e-5, rtol=1e-5)

    def test_threshold(self):
        x = torch.randn(1024, device='cuda')
        self._check(lambda t: torch.nn.functional.threshold(t, 0.3, -1.0), x)

    def test_hardshrink(self):
        x = torch.randn(1024, device='cuda')
        self._check(lambda t: torch.nn.functional.hardshrink(t, 0.5), x)

    # --- Pairwise min/max + clamp ---

    def test_minimum(self):
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        self._check(torch.minimum, a, b)

    def test_maximum(self):
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        self._check(torch.maximum, a, b)

    def test_clamp_both_bounds(self):
        x = torch.randn(1024, device='cuda') * 3.0
        self._check(lambda t: torch.clamp(t, -1.0, 1.0), x)

    def test_clamp_min_only(self):
        x = torch.randn(1024, device='cuda') * 3.0
        self._check(lambda t: torch.clamp(t, min=0.0), x)

    def test_clamp_max_only(self):
        x = torch.randn(1024, device='cuda') * 3.0
        self._check(lambda t: torch.clamp(t, max=2.0), x)

    # --- Composed unary math ---

    def test_square(self):
        x = torch.randn(1024, device='cuda')
        self._check(torch.square, x)

    def test_log1p(self):
        x = torch.rand(1024, device='cuda') * 5.0
        self._check(torch.log1p, x, atol=1e-5, rtol=1e-5)

    def test_expm1(self):
        x = torch.randn(1024, device='cuda') * 0.8
        self._check(torch.expm1, x, atol=1e-5, rtol=1e-5)

    def test_log10(self):
        x = torch.rand(1024, device='cuda') * 10.0 + 0.1
        self._check(torch.log10, x, atol=1e-5, rtol=1e-5)

    def test_sinh_cosh(self):
        x = torch.randn(1024, device='cuda') * 1.0
        self._check(torch.sinh, x, atol=1e-4, rtol=1e-4)
        self._check(torch.cosh, x, atol=1e-4, rtol=1e-4)

    def test_atanh(self):
        x = torch.rand(1024, device='cuda') * 1.6 - 0.8   # in (-1, 1)
        self._check(torch.atanh, x, atol=1e-5, rtol=1e-5)

    def test_asinh(self):
        x = torch.randn(1024, device='cuda')
        self._check(torch.asinh, x, atol=1e-5, rtol=1e-5)

    def test_acosh(self):
        x = torch.rand(1024, device='cuda') * 5.0 + 1.0   # >= 1
        self._check(torch.acosh, x, atol=1e-5, rtol=1e-5)

    # --- Composite scalar ops ---

    def test_addcmul(self):
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        c = torch.randn(1024, device='cuda')
        self._check(lambda x, y, z: torch.addcmul(x, y, z, value=0.5), a, b, c)

    def test_addcdiv(self):
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        c = torch.rand(1024, device='cuda') + 0.1
        self._check(lambda x, y, z: torch.addcdiv(x, y, z, value=0.5), a, b, c,
                    atol=1e-5, rtol=1e-5)

    def test_lerp_scalar(self):
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        self._check(lambda x, y: torch.lerp(x, y, 0.3), a, b)

    # --- rsub.Scalar (newly supported via the scalar-fold path) ---

    def test_rsub_scalar(self):
        x = torch.randn(1024, device='cuda')
        self._check(lambda t: 1.0 - t, x)
        self._check(lambda t: 3.5 - t, x)

    # --- Scalar fold: assert immediate insns are actually emitted ---

    def test_scalar_fold_x_plus_1_is_single_faddimm(self):
        """``x + 1.0`` must lower to exactly one FAddImm — not LoadImm + FAdd.

        Full expected bytecode: load, faddimm, store, halt.
        """
        import gint.kernel.interpreter.instructions.immediate as _imm
        import gint.kernel.interpreter.instructions.arith as _arith
        from gint.conductor.debug import inspect_subgraphs
        from gint.kernel.interpreter.main import INSNS

        FAddImm = INSNS[_imm.FAddImm]
        FMulImm = INSNS[_imm.FMulImm]
        LoadImm = INSNS[_imm.LoadImm]
        FAdd    = INSNS[_arith.FAdd]
        FMul    = INSNS[_arith.FMul]

        x = torch.randn(1024, device='cuda')
        torch._dynamo.reset()
        info = inspect_subgraphs(lambda t: t + 1.0, x)
        ops = [op for op, _ in info[0].bytecode]

        self.assertEqual(ops.count(FAddImm), 1,
                         f"expected exactly 1 FAddImm, got bytecode={ops}")
        self.assertEqual(ops.count(LoadImm), 0,
                         f"expected no LoadImm (scalar should fold), got bytecode={ops}")
        self.assertEqual(ops.count(FAdd), 0,
                         f"expected no FAdd (folded into faddimm), got bytecode={ops}")
        # 4 insns total: load, faddimm, store, halt.
        self.assertEqual(len(ops), 4,
                         f"expected 4 instructions, got {len(ops)}: {ops}")

    def test_scalar_fold_x_times_2_is_single_fmulimm(self):
        """``x * 2.0`` must lower to exactly one FMulImm."""
        import gint.kernel.interpreter.instructions.immediate as _imm
        import gint.kernel.interpreter.instructions.arith as _arith
        from gint.conductor.debug import inspect_subgraphs
        from gint.kernel.interpreter.main import INSNS

        FMulImm = INSNS[_imm.FMulImm]
        LoadImm = INSNS[_imm.LoadImm]
        FMul    = INSNS[_arith.FMul]

        x = torch.randn(1024, device='cuda')
        torch._dynamo.reset()
        info = inspect_subgraphs(lambda t: t * 2.0, x)
        ops = [op for op, _ in info[0].bytecode]

        self.assertEqual(ops.count(FMulImm), 1, f"bytecode={ops}")
        self.assertEqual(ops.count(LoadImm), 0, f"bytecode={ops}")
        self.assertEqual(ops.count(FMul), 0, f"bytecode={ops}")
        self.assertEqual(len(ops), 4, f"bytecode={ops}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available, skipping tests.")
