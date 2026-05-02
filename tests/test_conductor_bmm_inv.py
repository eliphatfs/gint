"""Conductor integration tests for the small-matrix bmm / inverse rewrite.

The conductor replaces ``aten.bmm.default`` and ``getitem(linalg_inv_ex, 0)``
with calls to ``gint.host.matrix.gint_bmm`` / ``gint_inv`` for matrices with
trailing dim N <= 4. These tests cover correctness across N and across batch
shapes, and verify that surrounding pointwise ops still fuse through the
gint codegen path.
"""

import unittest

import torch

import gint.conductor  # noqa: F401 — registers the "gint" backend
from gint.conductor import compile as gint_compile
from gint.conductor.debug import inspect_subgraphs
from tests import requires_gpu


@requires_gpu
class TestConductorBmmInv(unittest.TestCase):

    def _diag_dominant(self, B, N):
        torch.manual_seed(7)
        a = torch.randn(B, N, N, device='cuda', dtype=torch.float32)
        return a + 4.0 * torch.eye(N, device='cuda', dtype=torch.float32)

    # --- bmm correctness across N=1..4 ---

    def test_bmm_sizes(self):
        @gint_compile(options={"cuda_graphs": False})
        def fn(a, b):
            return torch.bmm(a, b)

        torch.manual_seed(42)
        with torch.inference_mode():
            for N in (1, 2, 3, 4):
                for B in (1, 32, 64, 1000):
                    a = torch.randn(B, N, N, device='cuda')
                    b = torch.randn(B, N, N, device='cuda')
                    out = fn(a, b)
                    ref = torch.bmm(a, b)
                    torch.testing.assert_close(
                        out, ref, atol=1e-5, rtol=1e-5,
                        msg=f"N={N} B={B}")

    # --- inverse correctness across N=1..4 ---

    def test_inv_sizes(self):
        @gint_compile(options={"cuda_graphs": False})
        def fn(a):
            return torch.linalg.inv(a)

        with torch.inference_mode():
            for N in (1, 2, 3, 4):
                for B in (1, 32, 1000):
                    a = self._diag_dominant(B, N)
                    out = fn(a)
                    ref = torch.linalg.inv(a)
                    torch.testing.assert_close(
                        out, ref, atol=1e-3, rtol=1e-3,
                        msg=f"N={N} B={B}")

    # --- multi-axis batch shape (e.g. (B1, B2, N, N)) ---

    def test_bmm_multi_batch(self):
        # ``a @ b`` on 4D tensors decomposes (expand → view → bmm → view).
        # The bmm node still has a concrete (B*B', N, N) shape, so the
        # rewrite catches it and the surrounding view/expand ops pass
        # through as metadata. ``gint_compile`` scopes dynamo to static
        # shapes so the per-shape recompile path stays concrete.
        @gint_compile(options={"cuda_graphs": False})
        def fn(a, b):
            return a @ b

        torch.manual_seed(123)
        with torch.inference_mode():
            for N in (2, 3, 4):
                a = torch.randn(8, 5, N, N, device='cuda')
                b = torch.randn(8, 5, N, N, device='cuda')
                out = fn(a, b)
                ref = a @ b
                torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # --- bmm fused into a pointwise pre/post chain (eager fallback for the
    #     bmm itself; gint codegen for the surrounding ops) ---

    def test_bmm_with_pointwise(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(a, b):
            return torch.bmm(a, b).relu() + 1.0

        torch.manual_seed(1)
        with torch.inference_mode():
            a = torch.randn(64, 4, 4, device='cuda')
            b = torch.randn(64, 4, 4, device='cuda')
            out = fn(a, b)
            ref = torch.bmm(a, b).relu() + 1.0
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_inv_with_pointwise(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(a):
            return torch.linalg.inv(a) * 2.0 + 1.0

        with torch.inference_mode():
            a = self._diag_dominant(64, 4)
            out = fn(a)
            ref = torch.linalg.inv(a) * 2.0 + 1.0
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    # --- N > 4 must NOT be rewritten (falls back to torch.bmm eagerly) ---

    def test_bmm_large_N_not_rewritten(self):
        @torch.compile(backend="gint", options={"cuda_graphs": False})
        def fn(a, b):
            return torch.bmm(a, b)

        torch.manual_seed(2)
        with torch.inference_mode():
            a = torch.randn(8, 5, 5, device='cuda')
            b = torch.randn(8, 5, 5, device='cuda')
            out = fn(a, b)
            ref = torch.bmm(a, b)
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # --- multi-shape recompile: dynamo's auto-dynamic promotion is
    #     disabled at backend registration, so the second compile gets
    #     concrete ints rather than SymInts. ---

    def test_recompile_across_shapes(self):
        @gint_compile(options={"cuda_graphs": False})
        def fn_pw(a, b):
            return a + b * 2.0

        @gint_compile(options={"cuda_graphs": False})
        def fn_bmm(a, b):
            return torch.bmm(a, b) + 1.0

        torch.manual_seed(7)
        with torch.inference_mode():
            for N in (3, 4, 5):
                a = torch.randn(8, 5, N, N, device='cuda')
                out = fn_pw(a, a)
                torch.testing.assert_close(out, a + a * 2.0)
            for N in (2, 3, 4):
                a = torch.randn(40, N, N, device='cuda')
                b = torch.randn(40, N, N, device='cuda')
                out = fn_bmm(a, b)
                ref = torch.bmm(a, b) + 1.0
                torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_explicit_dynamic_raises(self):
        # ``dynamic=True`` is a per-compile override that wins over
        # ``gint_compile``'s scoped patch — dynamo still produces SymInts
        # and we surface a clear error.
        @gint_compile(dynamic=True, options={"cuda_graphs": False})
        def fn(a, b):
            return a + b

        with torch.inference_mode():
            a = torch.randn(8, 5, 4, 4, device='cuda')
            with self.assertRaisesRegex(RuntimeError, "SymInt"):
                fn(a, a)

    # --- subgraph inspection: surrounding pointwise ops still go through
    #     gint codegen even when the bmm itself is eager-fallback ---

    def test_subgraphs_post_rewrite(self):
        def fn(a, b):
            return torch.bmm(a, b) + 1.0

        torch.manual_seed(3)
        a = torch.randn(64, 4, 4, device='cuda')
        b = torch.randn(64, 4, 4, device='cuda')
        with torch.inference_mode():
            sgs = inspect_subgraphs(fn, a, b)
        # Exactly one gint subgraph for the trailing pointwise add.
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'pointwise')


if __name__ == '__main__':
    unittest.main()
