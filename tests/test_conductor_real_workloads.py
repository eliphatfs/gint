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

    def test_per_batch_pre_reduction_fusion(self):
        """A per-batch op (relu of a [M,1] tensor) running alongside a
        per-chunk reduction (`(n*L).sum(-1)`) should fuse into a SINGLE
        kernel. The pre_prefix splits into:
          - per_batch = [relu]  → Phase 0, runs once before chunk loop
                                  with stride-0 inner loads / size-1 store
          - per_chunk = [mul_1] → Phase 1, runs inside the chunk loop
        Mirrors the geometry_smith middle subgraph that previously needed
        two kernels (pointwise + standalone reduction)."""
        def fn(scalar_in, n, L):
            relu_out = torch.relu(scalar_in)
            sum_out  = (n * L).sum(-1, keepdim=True)
            return relu_out, sum_out

        torch.manual_seed(0)
        scalar_in = torch.randn(4096, 1, device='cuda')
        n = torch.randn(4096, 3, device='cuda')
        L = torch.randn(4096, 3, device='cuda')

        with torch.no_grad():
            relu_ref, sum_ref = fn(scalar_in, n, L)
            compiled = torch.compile(fn, backend="gint",
                                     options={"cuda_graphs": False})
            relu_got, sum_got = compiled(scalar_in, n, L)
        torch.testing.assert_close(relu_got, relu_ref, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(sum_got, sum_ref, atol=1e-4, rtol=1e-4)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, scalar_in, n, L)
        # Single fused-reduction subgraph covering relu + n*L + sum.
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'fused-reduction')
        node_names = [nd.name for nd in sgs[0].nodes]
        self.assertIn('relu', node_names)
        self.assertIn('mul', node_names)
        self.assertIn('sum_1', node_names)

    def test_cse_dedups_duplicate_subexpressions(self):
        """torch.fx's CSE pass should run before partitioning, so two
        independent calls to a helper that internally computes the same
        expression collapse into a single computation.

        Mirrors the geometry_smith pattern: two ``schlick_ggx`` calls
        share ``roughness * roughness``, ``/ 2``, and ``1 - k`` —
        without CSE the second call recomputes them all."""
        def schlick_ggx(n_dot_x, roughness):
            a = roughness
            k = (a * a) / 2.0
            return n_dot_x / (n_dot_x * (1.0 - k) + k)

        def fn(x, y, roughness):
            return schlick_ggx(x, roughness) + schlick_ggx(y, roughness)

        torch.manual_seed(0)
        x = torch.rand(1024, 1, device='cuda') + 0.1
        y = torch.rand(1024, 1, device='cuda') + 0.1
        r = torch.rand(1024, 1, device='cuda') * 0.9 + 0.1

        ref = fn(x, y, r)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(x, y, r)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, x, y, r)
        # Without CSE the chain would be 13 nodes (6+6+1 = two schlick_ggx
        # bodies + final add). With CSE the three shared nodes
        # (r*r, /2, 1-k) collapse → 10 nodes.
        total_nodes = sum(len(sg.nodes) for sg in sgs)
        self.assertLessEqual(total_nodes, 10,
                             f"CSE didn't run — got {total_nodes} nodes, expected ≤ 10")

    def test_register_spill_fuses_long_chain(self):
        """A 14-node pointwise chain that exceeds the 8-slot stack must
        fuse into a single subgraph by spilling long-lived multi-user
        intermediates to virtual registers (l12 variant). Mirrors the
        post-reduction tail of ``geometry_smith`` (two ``schlick_ggx``
        calls + final multiply) where ``relu_1``, ``k₁``, ``k₂`` and
        the partial ggx results need to live across many ops."""
        def schlick_ggx(n_dot_x, roughness):
            a = roughness
            k = (a * a) / 2.0
            return n_dot_x / (n_dot_x * (1.0 - k) + k)

        def fn(n_dot_v, n_dot_L, roughness):
            relu_v = torch.relu(n_dot_v)
            relu_L = torch.relu(n_dot_L)
            return schlick_ggx(relu_L, roughness) * schlick_ggx(relu_v, roughness)

        torch.manual_seed(0)
        nv = torch.randn(4096, 1, device='cuda')
        nL = torch.randn(4096, 1, device='cuda')
        r  = torch.rand(4096, 1, device='cuda') * 0.9 + 0.1
        ref = fn(nv, nL, r)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(nv, nL, r)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, nv, nL, r)
        # All 14 chain ops should fuse into a single subgraph.
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].kind, 'pointwise')
        # The bytecode must use FStoreReg / FLoadReg, otherwise we're
        # falling back to global-memory spills.
        from gint.host.analyzer import analyze_bytecode
        stats = analyze_bytecode(sgs[0].bytecode)
        self.assertGreaterEqual(stats.max_reg_idx, 0)

    def test_long_pointwise_chain_flush(self):
        """Pointwise chain longer than max_stack — partitioner must flush
        the accumulated valid prefix as a subgraph instead of dropping
        every node when the next one fails the stack-fits check.

        Regression: ``GraphPartitioner.partition`` overwrote ``current``
        with ``[node]`` on the rejection path without first appending
        the existing ``current`` to ``subgraphs``, silently dropping
        every intermediate node into the eager-fallback path."""
        def fn(x, y):
            # 12 dependent pointwise ops on the same shape — at some
            # point the candidate exceeds MAX_STACK=8 and the partitioner
            # must split into two valid subgraphs (not lose the prefix).
            a = x + y
            b = a * 2.0
            c = b - 1.0
            d = c / 3.0
            e = d + y
            f = e * x
            g = f - y
            h = g / 2.0
            i = h + 1.0
            j = i * y
            k = j - x
            return k + 1.0

        torch.manual_seed(0)
        x = torch.randn(64, 16, device='cuda')
        y = torch.randn(64, 16, device='cuda')
        ref = fn(x, y)
        compiled = torch.compile(fn, backend="gint", options={"cuda_graphs": False})
        got = compiled(x, y)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        torch._dynamo.reset()
        sgs = inspect_subgraphs(fn, x, y)
        # Every FX node must end up in a subgraph (none dropped to eager).
        # The chain produces 12 elementwise nodes; the partitioner may
        # split them into multiple subgraphs but the union must cover all.
        all_nodes = set()
        for sg in sgs:
            for n in sg.nodes:
                all_nodes.add(n.name)
        # Each elementwise op must be covered by some subgraph.
        self.assertGreaterEqual(len(all_nodes), 12)


if __name__ == '__main__':
    unittest.main()
