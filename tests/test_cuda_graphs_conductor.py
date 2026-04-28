import torch
import unittest
from tests import requires_gpu
from gint.conductor import register_backend
from gint.conductor import backend as backend_mod


@requires_gpu
class TestConductorCudaGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            register_backend("gint_graphs", cuda_graphs=True)
        except Exception:
            pass

    def test_flag_resolution(self):
        import os
        self.assertFalse(backend_mod._resolve_cuda_graphs(None))
        self.assertFalse(backend_mod._resolve_cuda_graphs(False))
        self.assertTrue(backend_mod._resolve_cuda_graphs(True))
        os.environ['GINT_CUDA_GRAPHS'] = '1'
        try:
            self.assertTrue(backend_mod._resolve_cuda_graphs(None))
            self.assertFalse(backend_mod._resolve_cuda_graphs(False))
        finally:
            del os.environ['GINT_CUDA_GRAPHS']

    def test_pointwise_repeated_calls_fresh_inputs(self):
        @torch.compile(backend="gint_graphs")
        def fn(x, y):
            return torch.relu(x) + y * 2.0

        for _ in range(5):
            x = torch.randn(1024, device='cuda')
            y = torch.randn(1024, device='cuda')
            torch.testing.assert_close(fn(x, y), torch.relu(x) + y * 2.0)

    def test_intermediate_tensor_across_replays(self):
        @torch.compile(backend="gint_graphs")
        def fn(a, b, c):
            return (a + b) * c

        for _ in range(5):
            a = torch.randn(2048, device='cuda')
            b = torch.randn(2048, device='cuda')
            c = torch.randn(2048, device='cuda')
            torch.testing.assert_close(fn(a, b, c), (a + b) * c)

    def test_broadcast_under_graphs(self):
        @torch.compile(backend="gint_graphs")
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
