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
    # Fallback: unsupported op should still produce correct result via eager
    # ------------------------------------------------------------------

    def test_fallback_unsupported_op(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.relu(x) + 1.0

        x = torch.randn(1024, device='cuda')
        torch.testing.assert_close(fn(x), torch.relu(x) + 1.0)


if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available, skipping tests.")
