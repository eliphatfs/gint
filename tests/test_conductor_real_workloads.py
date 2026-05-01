"""Conductor end-to-end correctness tests on realistic multi-op workloads
extracted from real codebases.  These exercise partitioning, reduction
fusion, and broadcast in combinations that the unit-style cases in
``test_conductor_backend.py`` don't cover.  Add new functions here rather
than inflating the unit-test module."""

import torch
import unittest
from tests import requires_gpu
import gint.conductor  # noqa: F401  (auto-registers "gint" / "gint-no-cuda-graph")
from gint.conductor.debug import inspect_subgraphs


@requires_gpu
class TestRealWorkloads(unittest.TestCase):

    def test_geometry_smith_brdf(self):
        """diffrp/utils/light_transport.py:geometry_smith — Smith's
        masking-shadowing function for GGX BRDF.  Two parallel inner-dim
        sum-reductions (dot products), each followed by relu and a
        Schlick-GGX rational, joined by a final pairwise multiply.

        Regression: the conductor's post-reduction fusion was greedily
        pulling the *second* dot product's pre-reduction `n*L` mul into
        the *first* reduction's fused subgraph (because n,L last-dim
        matched reduction_size=3), then silently dropping the relu
        scalar that downstream subgraphs depended on."""
        def schlick_ggx(n_dot_x, roughness):
            a = roughness
            k = (a * a) / 2.0
            return n_dot_x / (n_dot_x * (1.0 - k) + k)

        def geometry_smith(n, v, L, roughness):
            n_dot_v = torch.relu((n * v).sum(-1, keepdim=True))
            n_dot_L = torch.relu((n * L).sum(-1, keepdim=True))
            ggx2 = schlick_ggx(n_dot_v, roughness)
            ggx1 = schlick_ggx(n_dot_L, roughness)
            return ggx1 * ggx2

        torch.manual_seed(7)
        M = 4096
        n = torch.randn(M, 3, device='cuda')
        v = torch.randn(M, 3, device='cuda')
        L = torch.randn(M, 3, device='cuda')
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        L = L / L.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        r = torch.rand(M, 1, device='cuda') * 0.9 + 0.1

        ref = geometry_smith(n, v, L, r)
        compiled = torch.compile(
            geometry_smith, backend="gint", options={"cuda_graphs": False})
        got = compiled(n, v, L, r)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)

    # ------------------------------------------------------------------
    # Small-N fused reduction WITH broadcast suffix
    # ------------------------------------------------------------------
    # The new 2dt/2dw tile path needs Phase 2 (per-chunk broadcast suffix)
    # to load external globals at chunk-offset positions and store outputs
    # back per-chunk.  These cases force `_select_reduction_tiling` to pick
    # 2dt (16 ≤ N < 128) or 2dw (N < 16) AND have a non-empty broadcast
    # suffix — the path that was unsupported until the Phase-2 generalization.

    def test_mean_subtract_small_N_2dt(self):
        """`x - mean(x, keepdim=True)` at N=64 → 2dt path with B=4 batches/warp.
        broadcast_suffix = [sub] (consumes scalar mean + external x)."""
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        torch.manual_seed(0)
        x = torch.randn(32, 64, device='cuda')
        ref = fn(x)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(x)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        # Confirm fusion: a single subgraph for the whole expression.
        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, x)
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'fused-reduction')

    def test_mean_subtract_small_N_2dw(self):
        """`x - mean(x, keepdim=True)` at N=8 → 2dw path with B=32 batches/warp."""
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        torch.manual_seed(0)
        x = torch.randn(64, 8, device='cuda')
        ref = fn(x)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(x)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, x)
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'fused-reduction')

    def test_rms_norm_small_N_2dt(self):
        """RMSNorm with small reduction dim — exercises pre_prefix (x*x),
        scalar_prefix (+eps, rsqrt), and broadcast_suffix (*x, *w) all
        together on the 2dt tile."""
        def rms_norm(x, w):
            rstd = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
            return x * rstd * w

        torch.manual_seed(0)
        x = torch.randn(32, 64, device='cuda')
        w = torch.randn(64, device='cuda')
        ref = rms_norm(x, w)
        compiled = torch.compile(rms_norm, backend="gint", options={"cuda_graphs": False})
        got = compiled(x, w)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(rms_norm, x, w)
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'fused-reduction')

    def test_rms_norm_small_N_2dw(self):
        """RMSNorm at N=8 — exercises 2dw tile with all three phases."""
        def rms_norm(x, w):
            rstd = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
            return x * rstd * w

        torch.manual_seed(0)
        x = torch.randn(128, 8, device='cuda')
        w = torch.randn(8, device='cuda')
        ref = rms_norm(x, w)
        compiled = torch.compile(rms_norm, backend="gint", options={"cuda_graphs": False})
        got = compiled(x, w)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(rms_norm, x, w)
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'fused-reduction')

    def test_normalize_small_N_post_only(self):
        """`x / sum(x, keepdim=True)` at small N — post-only fusion (no
        pre_prefix), broadcast_suffix consumes external x."""
        def fn(x):
            return x / torch.sum(x, dim=-1, keepdim=True)

        torch.manual_seed(0)
        x = torch.rand(32, 32, device='cuda') + 0.1  # positive, avoid div-by-zero
        ref = fn(x)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(x)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, x)
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'fused-reduction')

    def test_partial_last_tile_small_N(self):
        """Batch count not a multiple of B (32 in 2dw mode). The kernel's
        per-warp shape clamping should mask the remainder warps so OOB
        batches don't get spurious values."""
        def fn(x):
            return x - torch.mean(x, dim=-1, keepdim=True)

        torch.manual_seed(0)
        # 33 batches × 8 inner: last warp covers batch 32 only (1 of 32).
        x = torch.randn(33, 8, device='cuda')
        ref = fn(x)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(x)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
