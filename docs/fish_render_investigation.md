# diffrp fish render — gint torch.compile investigation

Handoff doc for the diffrp fish-render benchmark under `gint.conductor.compile`.
Captures findings as of 2026-05-03. Covers (1) a CUDA-graph correctness fix,
(2) a steady-state performance profile, and (3) what to look at next.

## TL;DR

- Bench script: `tmp/render_fish_orbit.py` (run from `tmp/diffrp-tests/`).
- **Correctness**: fish render produces the correct image (psnr 120 dB vs
  eager) under `gint.conductor.compile`. Required a CUDA-graph dtype filter at
  `gint/conductor/backend.py:151` — see "Corruption fix" below.
- **Performance fixed-azim steady-state**: gint 118 fps, eager 145 fps,
  jit 160 fps, gint-no-cg 64 fps. CG gives ~85% speedup over no-cg; the
  remaining 18% gap to eager is dominated by Python dispatch overhead
  (dynamo + AOT + per-frame CG-wrapper), **not** gint kernel time.
- **Performance varying-azim**: gint 121 fps, eager 151 fps (80% of eager).
  Was 28 fps before the bench script's nvdiffrast monkey-patch — see
  "Varying-azim recompile cascade" below. Raising `cache_size_limit`
  the naïve way made it dramatically worse (1.4 fps); recompile-cost
  grows with cache size.

## Repro

```bash
# from project root, into the diffrp data tree
cd tmp/diffrp-tests/

# correctness + fixed-azim perf (steady state)
python ../render_fish_orbit.py --mode gint --fixed-azim \
    --warmup 3 --iters 20 --cmp-ref /tmp/ref_eager.pt
# → ~118 fps, psnr 120 dB

# varying-azim (recompile-dominated)
python ../render_fish_orbit.py --mode gint \
    --warmup 3 --iters 20 --cmp-ref /tmp/ref_eager.pt
# → ~28 fps, psnr 120 dB

# baselines
python ../render_fish_orbit.py --mode eager --fixed-azim ...     # ~145 fps
python ../render_fish_orbit.py --mode jit --fixed-azim ...       # ~160 fps
python ../render_fish_orbit.py --mode gint-no-cg --fixed-azim ... # ~64 fps

# generate reference once
python ../render_fish_orbit.py --mode eager --fixed-azim --save-ref /tmp/ref_eager.pt
```

The reference image lives at `/tmp/ref_eager.pt`. The fp32 vs eager
difference of `7.129e-05` is a normal kernel-fma difference.

## Corruption fix

### Symptom

Without the dtype filter, the fish render produces a corrupt image:
`max_abs=2.209, psnr_ldr=20.97 dB` (badly wrong on ~16% of pixels).

### What it actually is

Drift trace (run via `GINT_CG_SKIP_INT=0`, instrumenting the wrapper to
clone+save each output and re-check it after subsequent frames run)
shows that **previously-returned cloned outputs from frames 37/38/40/41
get overwritten while still alive in Python**. Same address, refcounts
> 0, content drifts each iteration. So a captured CUDA graph for some
later frame is writing to an address that we already returned to the
caller as a "safe" clone.

### Fix

`gint/conductor/backend.py:151` — skip CG-wrapping for any AOT frame
whose example_inputs include a non-fp32 tensor. Override with
`GINT_CG_SKIP_INT=0`.

The skipped frames are exactly the ones whose AOT graphs contain
`aten.index.Tensor` advanced indexing on int64 indices — diffrp's
rasterization-style `compose_layers` + per-pixel-triangle-id pipelines
emit them via `torch.gather`. They run via the eager fallback path
without CG, which is correct (no graph capture) but loses CG speedup
on those frames.

### Investigation status (so the next person doesn't repeat it)

We do **not** have a clean root cause; we have a conservative filter that
masks the bug.

Things ruled out by minimal repros (none reproduced the corruption):
- Pure-gint multi-frame (50 captured graphs): clean.
- Mixed gint+eager+`aten.index.Tensor`+int64 across 50 frames: clean.
- Same as above with large fp32 intermediates that match diffrp's IBL
  shapes (512×512×3, 5×256×512×3, 94003×3, etc.): clean.
- Initial dinfo-aliasing hypothesis (single shared `program._cu` device
  buffer overwritten across captures): wrong. `gint/host/cuda/executor_impl.py`
  is fine — each conductor subgraph has its own `GintCompiledSubgraph`
  instance, so `_cu` is never shared across captures.
- `make_graphed_callables`-style shared mempool (one `graph_pool_handle()`
  for all captures): no help.
- Replacing `o.clone()` with `torch.empty(...).copy_(o)` to force the
  output buffer into the regular pool: no help; drift still happens at
  the same Python addresses.
- Enlarging `torch._dynamo.config.cache_size_limit` to 64: catastrophic
  (1.4 fps). Recompile cost grows; doesn't avoid the bug either.

What we know for sure:
- Per-frame eager-vs-wrapped diff is `0` for every individual frame.
  Errors only emerge from **cross-frame interaction** during the full
  diffrp pipeline.
- The corrupting writers in our trace were frames 37/38/40/41 — the
  IBL-combine + 5×256×512×3 reduce + the big 18-input pointwise fuse
  near the end of `pbr()`. Their captures land at addresses that
  somehow alias outputs we already returned for OTHER, unrelated
  earlier frames.
- This looks like an interaction between AOT autograd's mixed
  eager-fallback / gint runtime and torch's CUDA-graph mempool
  bookkeeping, not a gint executor bug.

If you pursue this further: instrument the captured pool's allocations
via `torch.cuda.memory._snapshot()` or `_record_memory_history()` to see
whether captured-pool VAs really overlap regular-pool addresses, and
whether the OBSERVED drift addresses are actually inside any captured
pool's reserved range.

### Diagnostic env vars

- `GINT_CG_SKIP_INT=0` — disable the dtype filter (reproduces corruption).
- `GINT_CG_DUMP_AOT_INT64=1` — print every AOT FX graph that has an int64
  input. (Wired at `gint/conductor/backend.py:89`.)

There are no other env-var hooks in the current backend; the
investigation-time hooks (`GINT_CG_DEBUG_OUTPUT`, `GINT_CG_LOG`,
`GINT_CG_DUMP_FX`, `GINT_CG_DEBUG_MAX_FRAMES`, `GINT_CG_ONLY_FRAMES`)
were removed when the dtype filter shipped. Re-add from git history if
you need them.

## Varying-azim recompile cascade

Without intervention, varying-azim renders run 4× slower than fixed-azim
(28 fps vs 118 fps). Cause: dynamo recompiles `nvdiffrast.torch.ops:257`'s
`_rasterize_func.forward` per `peeling_idx` value (the C extension
`PyCapsule.rasterize_fwd_cuda` isn't traceable, so dynamo treats each
peel pass as a fresh frame). After 8 such recompiles the cache_size_limit
trips and dynamo recompiles into the slow eager-fallback dispatch path.
Worse, the same fires for `sample2d`, `interpolate_ex`, `sample`, `fsa`,
`fma`, `ones_like_vec`, `cross`, `normalized`, `float3` because those
specialize on the per-pixel triangle/uv tensor shapes which DO vary
when the camera moves (visible/culled triangle counts change).

**Fix** (currently applied as a monkey-patch in `tmp/render_fish_orbit.py`,
`install_nvdiffrast_patch`): wrap `_rasterize_func.forward` in
`torch._dynamo.disable`. dynamo treats it as a graph-break leaf instead
of trying to trace through, the per-`peeling_idx` recompile loop stops,
and downstream specialization on rasterize-output shapes also stops
(because dynamo doesn't see them as graph-internal anymore). Zero
`cache_size_limit` warnings remain after the patch.

Result:
- gint varying-azim: 28 fps → 121 fps (4.3×)
- gint fixed-azim: unchanged (no recompiles to begin with)

The patch is safe for all modes — eager and jit don't run dynamo so the
wrap is a no-op for them; the function still calls the original C
extension when invoked. The patch should ideally land upstream in
nvdiffrast (proper fix: register `rasterize_fwd_cuda` as a
`torch.library.custom_op` with a fake/meta implementation so dynamo can
trace THROUGH it instead of around it — that would also let CG capture
the rasterize call alongside surrounding gint work, eliminating one of
the graph-break boundaries).

## Steady-state performance profile

Profile via `torch.profiler` with `schedule(skip_first=2, wait=1, warmup=2,
active=20, repeat=1)` so only fully-warm iters get aggregated. The
script lives at `/tmp/profile_fish_steady.py` (not in git — easy to
regenerate; uses the same render setup as the bench).

**Important**: the naïve `with profile(): for _ in range(N): render()`
pattern produces measurement artifacts on iter 1 (e.g. an earlier run
showed 9.87 ms / 10 calls of `aten::linalg_solve_ex` from one-shot
runtime metadata work that does **not** repeat in steady state).
Always use `schedule(...)` with non-zero `skip_first` and `warmup`.

### Per-iter breakdown (20 iters aggregated)

CUDA time (4.4 ms/iter total):

| op | self_cuda | calls/iter |
|---|---|---|
| geval_s7 (gint kernel) | 0.81 ms (18%) | 110 |
| aten::copy_ | 0.83 ms (19%) | 356 |
| Memcpy DtoD | 0.50 ms (11%) | 165 |
| grid_sampler_2d | 0.38 ms (9%) | 49 |
| elementwise kernels (sum) | ~1.3 ms (~30%) | — |

CPU dispatch (per iter):

| | self_cpu/iter |
|---|---|
| Torch-Compiled Region | 5.4 ms |
| cudaLaunchKernel | 1.92 ms (1004 launches/iter) |
| cudaMemcpyAsync | 0.56 ms (196/iter) |
| TorchDynamo Cache Lookup | 0.43 ms |

Wall: 8.4 ms/iter without the profiler attached, 14 ms with (profiler
adds ~5 ms overhead per step). gint's own compute (geval_s7) is 18%
of GPU time. The system is **CPU-dispatch bound**: 1004 kernel
launches per iter, only 110 of which are gint kernels. The other ~890
are eager-fallback ops captured into the graph.

### Things that are NOT in the profile (despite first-iter artifacts)

- `aten::linalg_inv_ex` / `aten::linalg_solve_ex` / `aten::linalg_lu_solve`
  — only appear with `<10 µs/iter` if at all. They're not steady-state
  costs. The earlier ~1 ms/iter was first-iter pollution.
- `gint_inv` (our N≤4 inverse rewrite at `special_ops.py:78`) is correctly
  catching every `linalg_inv_ex` call that goes through the conductor.
  diffrp itself only uses `numpy.linalg.inv` (CPU) for the view matrix
  and `torch.linalg.inv_ex` in `raycaster.py` (which the
  `SurfaceDeferredRenderSession` path does not exercise).

## Optimization opportunities (priority order)

1. **Reduce launch count** — 890 eager-fallback launches/iter is the
   biggest leverage. Identify the highest-call-count fallback ops and
   add them to `op_registry.py`. Likely candidates from the profile:
   `aten::index` (1120 calls / 20 iter = 56/iter), `aten::_index_put_impl_`
   (32/iter), `aten::cat` (28/iter), `aten::nonzero` (8/iter),
   `aten::le` (26/iter), `aten::remainder` (32/iter). Some of these
   (`index`, `_index_put_impl_`, `nonzero`) involve int64 indices and
   will keep dropping into the dtype-filtered fallback path until the
   corruption is properly fixed (see above).
2. **Reduce copy_ + Memcpy DtoD volume** — 1.33 ms combined. Sources:
   - CG wrapper input copy (`s_in.copy_(a)` per input per call) at
     `gint/conductor/backend.py:200`.
   - CG wrapper output clone (`o.clone()` per output) at
     `gint/conductor/backend.py:203`.
   - Conductor's `_run_subgraph` allocates each output via `torch.empty`
     then writes into it — the empty itself doesn't memcpy but it
     drives a lot of the 196 cudaMemcpyAsync calls (one per
     gint kernel for the dinfo upload).
3. **Fold dinfo upload** — `gint/host/cuda/executor_impl.py:87-89` does
   one `cuMemcpyHtoDAsync` per `execute()` call to upload base_ptrs.
   Per-program dinfo is currently re-uploaded on every call even when
   addresses are stable; track address generation and skip when
   unchanged. Worth ~196 small memcpys/iter.
4. The remaining 5.4 ms Python dispatch (Torch-Compiled Region) is
   mostly dynamo + AOT runtime overhead; hard to reduce without
   restructuring how the backend interfaces with AOT.

## Key file/line references

- `gint/conductor/backend.py:151` — dtype filter that skips CG wrap for
  non-fp32-input frames.
- `gint/conductor/backend.py:163-213` — `_direct_cudagraph_wrap`: the
  inference-only CG wrapper. Per-call: input copy_ + replay + output
  clone.
- `gint/conductor/special_ops.py:70-88` — `linalg_inv_ex` rewrite (N≤4).
  Does NOT cover `linalg_solve_ex`; not currently a measured cost.
- `gint/conductor/compiler.py:2569-2620` — `_run_subgraph`: allocates
  fresh `torch.empty` outputs per call and dispatches the gint program.
- `gint/host/cuda/executor_impl.py:51-91` — `CudaExecutor.execute`:
  per-call dinfo upload. Sync `cuMemcpyHtoD` during capture, async
  `cuMemcpyHtoDAsync` outside.
- `tests/test_cuda_graphs_conductor.py` — regression tests for CG
  output integrity. `test_multi_frame_outputs_held_across_replays` is
  the closest synthetic test to the corruption case (passes even
  though the fish render fails — which is what makes the bug hard).
- `tmp/render_fish_orbit.py` — the bench. Use `--mode {eager,jit,gint,gint-no-cg}`.
