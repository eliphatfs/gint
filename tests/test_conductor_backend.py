import torch
import unittest
from gint.conductor import register_backend


class TestConductorBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            register_backend("gint_test")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Original tests
    # ------------------------------------------------------------------

    def test_simple_arithmetic(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x + y * 2.0

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y * 2.0)

    def test_partitioning_limit(self):
        @torch.compile(backend="gint_test")
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
        @torch.compile(backend="gint_test")
        def fn(x):
            res = x
            for _ in range(14):
                res = res + 1.0
            return res + 1.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), fn.__wrapped__(x))

    def test_mixed_shapes(self):
        @torch.compile(backend="gint_test")
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
        @torch.compile(backend="gint_test")
        def fn(x):
            return -x

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), -x)

    def test_unary_sqrt(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sqrt(x)

        x = torch.rand(1024, device='cuda') + 0.1
        torch.testing.assert_close(fn(x), torch.sqrt(x))

    def test_unary_exp_log(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.log(torch.exp(x))

        x = torch.randn(1024, device='cuda') * 0.5
        torch.testing.assert_close(fn(x), x, atol=1e-5, rtol=1e-5)

    def test_unary_sin_cos(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.sin(x) + torch.cos(x), atol=1e-5, rtol=1e-5)

    def test_unary_abs(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.abs(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.abs(x))

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def test_relu(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.relu(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.relu(x))

    def test_gelu(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.nn.functional.gelu(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.nn.functional.gelu(x), atol=1e-5, rtol=1e-5)

    def test_silu(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.nn.functional.silu(x), atol=1e-5, rtol=1e-5)

    def test_leaky_relu(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.nn.functional.leaky_relu(x, negative_slope=0.1)

        x = torch.randn(1024, device='cuda')
        expected = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        torch.testing.assert_close(fn(x), expected)

    # ------------------------------------------------------------------
    # Comparisons and where
    # ------------------------------------------------------------------

    def test_comparison_gt(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return (x > y).float()

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x, y), (x > y).float())

    def test_where(self):
        # Use comparison result as float mask (0.0/1.0) to avoid bool-tensor complications.
        @torch.compile(backend="gint_test")
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
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.relu(x) * 2.0 + 1.0

        x = torch.randn(1024, device='cuda')
        expected = torch.relu(x) * 2.0 + 1.0
        torch.testing.assert_close(fn(x), expected)

    def test_gelu_chain(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.nn.functional.gelu(x + 0.5) - 0.1

        x = torch.randn(1024, device='cuda')
        expected = torch.nn.functional.gelu(x + 0.5) - 0.1
        torch.testing.assert_close(fn(x), expected, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def test_broadcast_bias_add(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x + y

        x = torch.randn(32, 128, device='cuda')
        y = torch.randn(128, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    def test_broadcast_scalar_tensor(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x * y

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1, device='cuda')
        torch.testing.assert_close(fn(x, y), x * y)

    def test_broadcast_both_expand(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x + y

        x = torch.randn(64, 1, device='cuda')
        y = torch.randn(1, 32, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    def test_broadcast_3d(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x + y

        x = torch.randn(4, 8, 64, device='cuda')
        y = torch.randn(64, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    def test_broadcast_with_activation(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return torch.relu(x + y)

        x = torch.randn(32, 128, device='cuda')
        y = torch.randn(128, device='cuda')
        torch.testing.assert_close(fn(x, y), torch.relu(x + y))

    def test_broadcast_same_shape(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x + y

        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x, y), x + y)

    # ------------------------------------------------------------------
    # Fallback: unsupported op should still produce correct result via eager
    # ------------------------------------------------------------------

    def test_fallback_unsupported_op(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.relu(x) + 1.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.relu(x) + 1.0)

    # ------------------------------------------------------------------
    # Metadata ops (view, unsqueeze, squeeze, expand, etc.)
    # ------------------------------------------------------------------

    def test_unsqueeze_squeeze_identity(self):
        """unsqueeze + squeeze cancel out; should fuse with surrounding ops."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return x.unsqueeze(0).squeeze(0) * 2.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), x * 2.0)

    def test_unsqueeze_fuse_with_relu(self):
        """relu + unsqueeze + add should fuse into one subgraph."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.relu(x).unsqueeze(0) + 1.0

        x = torch.randn(1024, device='cuda')
        expected = torch.relu(x).unsqueeze(0) + 1.0
        torch.testing.assert_close(fn(x), expected)

    def test_unsqueeze_broadcast_add(self):
        """unsqueeze enables broadcasting: (N,) → (1, N) + (M, N) → (M, N)."""
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x.unsqueeze(0) + y

        x = torch.randn(128, device='cuda')
        y = torch.randn(32, 128, device='cuda')
        expected = x.unsqueeze(0) + y
        torch.testing.assert_close(fn(x, y), expected)

    def test_expand_add(self):
        """expand broadcasts size-1 dim; fuses with add."""
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x.expand(32, 128) + y

        x = torch.randn(1, 128, device='cuda')
        y = torch.randn(32, 128, device='cuda')
        expected = x.expand(32, 128) + y
        torch.testing.assert_close(fn(x, y), expected)

    def test_view_pointwise(self):
        """view followed by pointwise on the new shape."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return x.view(32, 32) + 1.0

        x = torch.randn(1024, device='cuda')
        expected = x.view(32, 32) + 1.0
        torch.testing.assert_close(fn(x), expected)

    def test_metadata_standalone(self):
        """A metadata op alone (no fusion) should still produce correct result."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return x.unsqueeze(0) + 1.0

        x = torch.randn(512, device='cuda')
        expected = x.unsqueeze(0) + 1.0
        torch.testing.assert_close(fn(x), expected)


    # ------------------------------------------------------------------
    # Reduction ops (sum, mean)
    # ------------------------------------------------------------------

    def test_sum_1d(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=0)

        x = torch.randn(128, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=0), atol=1e-4, rtol=1e-4)

    def test_sum_innermost(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_sum_full_warp(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(16, 128, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_mean_innermost(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.mean(x, dim=-1)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.mean(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_sum_keepdim(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=-1, keepdim=True)

        x = torch.randn(32, 128, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1, keepdim=True), atol=1e-4, rtol=1e-4)

    def test_mean_keepdim(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.mean(x, dim=-1, keepdim=True), atol=1e-4, rtol=1e-4)

    def test_sum_large_dim(self):
        """Reduction dim > 128, requires multi-chunk unrolled loads."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(8, 512, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-3, rtol=1e-3)

    def test_sum_3d(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(4, 8, 32, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=-1), atol=1e-4, rtol=1e-4)

    def test_sum_non_innermost_fallback(self):
        """Non-innermost reduction should fall back to eager."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.sum(x, dim=0)

        x = torch.randn(32, 64, device='cuda')
        torch.testing.assert_close(fn(x), torch.sum(x, dim=0), atol=1e-4, rtol=1e-4)

    def test_mean_subtract(self):
        """Reduction + broadcast pointwise: x - mean(x, keepdim=True)."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(16, 64, device='cuda')
        expected = x - torch.mean(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-4, rtol=1e-4)

    def test_normalize_fused(self):
        """Fused: x / sum(x, keepdim=True)."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return x / torch.sum(x, dim=-1, keepdim=True)

        x = torch.rand(8, 32, device='cuda') + 0.1  # positive to avoid div-by-zero
        expected = x / torch.sum(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-4, rtol=1e-4)

    def test_mean_subtract_large(self):
        """Fused reduction+pointwise with multi-chunk (dim > 128)."""
        @torch.compile(backend="gint_test")
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        x = torch.randn(4, 256, device='cuda')
        expected = x - torch.mean(x, dim=-1, keepdim=True)
        torch.testing.assert_close(fn(x), expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available, skipping tests.")
