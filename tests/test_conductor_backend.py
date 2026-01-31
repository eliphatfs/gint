import torch
import unittest
from gint.conductor import register_backend

class TestConductorBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            register_backend("gint_test")
        except:
            pass

    def test_simple_arithmetic(self):
        # register_backend("gint_simple")
        @torch.compile(backend="gint_test")
        def fn(x, y):
            return x + y * 2.0
            
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        
        expected = x + y * 2.0
        actual = fn(x, y)
        
        torch.testing.assert_close(actual, expected)

    def test_partitioning_limit(self):
        # 9 tensors: 2 inputs (x, y) + 7 intermediate/outputs
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
        actual = fn(x, y)
        
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as e:
            print(f"Expected: {expected[:10]}")
            print(f"Actual: {actual[:10]}")
            print(f"Diff: {(actual - expected).abs()[:10]}")
            raise e

    def test_fallback_unsupported_op(self):
        @torch.compile(backend="gint_test")
        def fn(x):
            return torch.relu(x) + 1.0
            
        x = torch.randn(1024, device='cuda')
        expected = torch.relu(x) + 1.0
        actual = fn(x)
        
        torch.testing.assert_close(actual, expected)

    def test_mixed_shapes(self):
        @torch.compile(backend="gint_test")
        def fn(x, y):
            # x: [1024], y: [512]
            # This should force separate subgraphs or fallback
            return x + 1.0, y * 2.0
            
        x = torch.randn(1024, device='cuda')
        y = torch.randn(512, device='cuda')
        
        expected_x, expected_y = x + 1.0, y * 2.0
        actual_x, actual_y = fn(x, y)
        
        torch.testing.assert_close(actual_x, expected_x)
        torch.testing.assert_close(actual_y, expected_y)

    def test_many_ops_one_subgraph(self):
        # 16 tensors total if all were outputs, but only 2 external (1 input, 1 output)
        # Should fit in one subgraph now.
        @torch.compile(backend="gint_test")
        def fn(x):
            res = x
            for _ in range(14):
                res = res + 1.0
            return res + 1.0
            
        x = torch.randn(1024, device='cuda')
        expected = fn.__wrapped__(x)
        actual = fn(x)
        
        torch.testing.assert_close(actual, expected)

if __name__ == "__main__":
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available, skipping tests.")
