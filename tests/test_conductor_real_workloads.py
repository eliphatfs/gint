"""Conductor end-to-end correctness tests on realistic multi-op workloads
extracted from real codebases.  These exercise partitioning, reduction
fusion, and broadcast in combinations that the unit-style cases in
``test_conductor_backend.py`` don't cover.  Add new functions here rather
than inflating the unit-test module."""

import torch
import unittest
from tests import requires_gpu
import gint.conductor  # noqa: F401  (auto-registers "gint" / "gint-no-cuda-graph")


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


if __name__ == '__main__':
    unittest.main()
