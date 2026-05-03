import torch
import unittest
from tests import requires_gpu
import gint.conductor  # noqa: F401  (auto-registers "gint" / "gint-no-cuda-graph")


@requires_gpu
class TestConductorCudaGraphs(unittest.TestCase):
    def test_pointwise_repeated_calls_fresh_inputs(self):
        @torch.compile(backend="gint")
        def fn(x, y):
            return torch.relu(x) + y * 2.0

        for _ in range(5):
            x = torch.randn(1024, device='cuda')
            y = torch.randn(1024, device='cuda')
            torch.testing.assert_close(fn(x, y), torch.relu(x) + y * 2.0)

    def test_intermediate_tensor_across_replays(self):
        @torch.compile(backend="gint")
        def fn(a, b, c):
            return (a + b) * c

        for _ in range(5):
            a = torch.randn(2048, device='cuda')
            b = torch.randn(2048, device='cuda')
            c = torch.randn(2048, device='cuda')
            torch.testing.assert_close(fn(a, b, c), (a + b) * c)

    def test_broadcast_under_graphs(self):
        @torch.compile(backend="gint")
        def fn(x, y):
            return x + y

        for _ in range(3):
            x = torch.randn(32, 128, device='cuda')
            y = torch.randn(128, device='cuda')
            torch.testing.assert_close(fn(x, y), x + y)

    def test_multi_frame_outputs_held_across_replays(self):
        """When ``gint_compile`` wraps a function with several graph
        breaks, each frame becomes its own captured cuda graph. Without
        clone-on-output, the per-frame outputs returned to dynamo's
        runtime alias the captured pool addresses; later frames'
        captures + replays reuse those addresses and silently overwrite
        the held outputs (the diffrp fish render hit this — bisected
        to >=6 captures on RTX 4090). The wrapper must clone outputs
        out of the captured pool so callers see stable data.

        Regression: cycle two ``gint_compile``-wrapped functions
        through several calls, with the first's output read AFTER the
        second's wrapper has replayed (mimicking the held-output flow
        between dynamo frames). Both must keep returning correct values.
        """
        from gint.conductor import compile as gint_compile

        @gint_compile
        def f1(x):
            return x * 2.0 + 1.0

        @gint_compile
        def f2(y):
            return y - 0.5

        for k in range(8):
            x = torch.full((1024,), float(k), device='cuda')
            y = torch.full((1024,), float(k * 2), device='cuda')
            o1 = f1(x)
            o2 = f2(y)  # would alias o1 if clone is missing
            torch.cuda.synchronize()
            torch.testing.assert_close(o1, x * 2.0 + 1.0,
                                       msg=f"f1 output corrupted at iter {k}")
            torch.testing.assert_close(o2, y - 0.5,
                                       msg=f"f2 output corrupted at iter {k}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available, skipping tests.")
