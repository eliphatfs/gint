"""Calling gint kernels from inside upstream ``torch.compile`` must not
recurse into the bytecode-recording stack.

Without ``torch.compiler.disable`` on the executor entry points, dynamo
tries to trace through ``BaseExecutableProgram.__call__`` →
``executor.execute`` → ``program.get_program`` → the bytecode-recording
ContextVar set/reset, and dies with ``'NoneType' object has no attribute
'make_guard'`` (or similar internal errors).

These tests verify the workaround: a graph break at the kernel-call
boundary, so dynamo compiles the surrounding tensor code and runs the
gint call eagerly between compiled regions.
"""

import unittest

import torch

import gint  # noqa: F401
from gint.host.matrix import gint_bmm, gint_inv
from tests import requires_gpu


@requires_gpu
class TestTorchCompileCompat(unittest.TestCase):

    def setUp(self):
        torch._dynamo.reset()

    def test_torch_compile_around_gint_bmm(self):
        @torch.compile
        def fn(a, b):
            return gint_bmm(a, b) + 1

        a = torch.randn(8, 4, 4, device='cuda')
        b = torch.randn(8, 4, 4, device='cuda')
        out = fn(a, b)
        ref = (a @ b) + 1
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_torch_compile_around_gint_inv(self):
        @torch.compile
        def fn(a):
            return gint_inv(a) * 2

        a = torch.randn(8, 4, 4, device='cuda') + 4.0 * torch.eye(4, device='cuda')
        out = fn(a)
        ref = torch.linalg.inv(a) * 2
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_torch_compile_single_graph_break_per_call(self):
        """gint_bmm is decorated with ``@torch.compiler.disable`` so it
        contributes exactly one graph break — surrounding tensor code
        compiles into one graph, not two."""
        from torch._dynamo.utils import counters

        @torch.compile(backend='aot_eager')
        def fn(a, b):
            return gint_bmm(a, b) + 1

        a = torch.randn(8, 4, 4, device='cuda')
        b = torch.randn(8, 4, 4, device='cuda')

        counters.clear()
        fn(a, b)

        # Exactly one graph break, attributed to the disabled wrapper.
        self.assertEqual(sum(counters['graph_break'].values()), 1,
                         f"expected 1 graph break, got {dict(counters['graph_break'])}")
        self.assertEqual(counters['stats']['unique_graphs'], 1,
                         f"expected 1 compiled graph, got {counters['stats']['unique_graphs']}")

    def test_torch_compile_around_user_bytecode_kernel(self):
        """Any ``@bytecode``-decorated user kernel — not just the
        public matrix wrappers — must also be safely callable from
        inside ``torch.compile``. The workaround lives on
        ``BaseExecutableProgram.__call__`` so it covers all
        ``SugarProgram`` subclasses uniformly."""
        from gint.host.matrix import bmm4x4_kernel

        @torch.compile
        def fn(a, b):
            c = torch.empty_like(a)
            bmm4x4_kernel(a, b, c, grid_dim=1)
            return c + 1.0

        a = torch.randn(8, 16, device='cuda')
        b = torch.randn(8, 16, device='cuda')
        out = fn(a, b)

        # Reference: hand-roll the same 4x4 batched matmul.
        ref = (a.view(8, 4, 4) @ b.view(8, 4, 4)).reshape(8, 16) + 1.0
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_gint_compile_still_works(self):
        """Sanity: ``torch.compiler.disable`` on ``__call__`` must not
        break the gint conductor backend itself, which runs the kernel
        from inside its ``_run_eager`` callback (outside any dynamo
        trace) and also via the FX rewrite for ``aten.bmm``."""
        from gint.conductor import compile as gint_compile

        @gint_compile
        def fn(a, b):
            return torch.bmm(a, b) + 1

        a = torch.randn(8, 4, 4, device='cuda')
        b = torch.randn(8, 4, 4, device='cuda')
        out = fn(a, b)
        ref = torch.bmm(a, b) + 1
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
