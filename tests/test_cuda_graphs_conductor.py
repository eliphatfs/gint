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


if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available, skipping tests.")
