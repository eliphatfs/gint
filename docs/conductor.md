# Conductor (torch.compile backend)

Reference for the torch.compile integration. See `gint/conductor/` for source.

## Overview

- `gint/conductor/backend.py`: backend registration and entry point.
- `gint/conductor/compiler.py`: FX graph → bytecode conversion, graph partitioning, and broadcasting.
- `gint/conductor/debug.py`: `inspect_subgraphs(fn, *args)` runs `fn` through the gint backend and returns one `SubgraphInfo` per compiled subgraph (kind, FX nodes, bytecode, output shape, grid dim). Pair with `print_subgraphs` to dump the bytecode of each subgraph — the right tool when a torch.compile path looks slower than expected or you want to verify partitioning / fusion behavior.

Supported op surface (full list in `op_registry.py`; unsupported ops fall back to eager):
- Arithmetic: add, sub, mul, div, remainder, neg, abs, square, reciprocal, pow, atan2, rsub.Scalar
  - **Scalar fold**: `add/sub/mul/div` with a Python-scalar second arg lower to a single immediate insn (`faddimm` / `fmulimm`) instead of `LoadImm + fadd/fmul`. Driven by `OpDescriptor.arg_order` accepting a callable so the descriptor decides at emit time whether to push the scalar or skip it. `emit_op` takes a `num_pushed` count instead of relying on static `arity` so the codegen pop matches actual pushes.
- Comparisons + ternary: gt/lt/ge/le/eq/ne (Tensor + Scalar), `where`
- Transcendentals: sqrt, rsqrt, exp, exp2, expm1, log, log2, log10, log1p, sin/cos/tan, asin/acos/atan, sinh/cosh, atanh/asinh/acosh, erf
- Activations: relu, gelu, silu, leaky_relu, tanh, sigmoid, hardtanh, relu6, hardsigmoid, hardswish, softplus (beta=1), mish, elu, selu, threshold, hardshrink
- Pairwise + clamp: minimum, maximum, clamp/clamp_min/clamp_max (scalar + Tensor)
- Composite: addcmul, addcdiv, lerp.Scalar
- Metadata: view, unsqueeze, squeeze, expand, permute, transpose, t, slice
- Reductions (innermost dim only): sum, mean, prod, amax, amin

Partitioner constraints per subgraph: max 8 global tensor slots, max stack depth 8, broadcast-compatible shapes, all output globals must share the same iteration shape.

## Gint-Eligible Node Set (dtype fixed point)

- gint kernels read and write 4-byte float words. A node is gint-compileable only when **every** input boundary and the output boundary are floats — otherwise gint either picks up garbage from eager (e.g. an int64 index tensor reinterpreted as float) or hands garbage back.
- Two ops have legitimate non-float endpoints: `gt/lt/ge/le/eq/ne` produce bool, and `where.self` consumes a bool predicate. Internally, comparisons emit 0.0/1.0 and `where.self` reads via fselect — so the bool channel is fine **as long as both endpoints stay inside the same gint subgraph**. Crossing the eager boundary with a bool tensor would round-trip through bitwise reinterpretation.
- `op_registry.compute_gint_node_set(graph)` resolves this with a fixed-point pass:
  1. Seed with every registered, `check_fn`-passing call_function node.
  2. Drop any node whose dtype contract fails *given the current candidate set* (`_node_dtypes_compatible`). The bool channel is allowed only when every comparison user is a candidate `where.self` and every `where.self` cond comes from a candidate comparison.
  3. Iterate. A dropped `where` poisons its comparison feeders, which may in turn poison other shared-where consumers, until the set is stable.
- The result is cached on `graph._gint_node_set`. `get_op_descriptor(node)` consults the set when a partition is in flight, falling back to a single-node check (with an empty allow-set) when called outside the partitioner — so direct codegen / unit-test paths still get conservative answers.

## Output-Driven Iteration Shape

- The kernel iterates over the *output* shape, not over a global broadcast of every tensor. Letting an input drive the iteration would cause a kernel that reads `view(small) → big` to iterate `big`'s cells while writing to a buffer sized for `small`, smashing the same address with different values.
- During growth, `GraphPartitioner.partition` rejects a candidate node whose addition would give the subgraph two outputs with different shapes (`out_shapes` collected from `global_nodes ∩ scheduled`). The check fires before `_compute_broadcast_plan` so we never propagate an inflated iteration shape into stride computation.
- `_compile_subgraph` then derives `output_shape` from the broadcast of *output* globals only. Inputs are validated against that shape via `_compute_broadcast_plan` — small inputs broadcast UP into the iteration via stride-0 padding, but no input is allowed to grow the iteration beyond what every output buffer can hold.
- The "outputs must agree" rule splits patterns like `relu(scalar_in) + (n*L).sum(-1)` into separate pointwise + reduction subgraphs at the partition layer; the merger phase (see *Fused Reduction+Broadcast+Pointwise*) restores the per-batch + per-chunk fusion when a reduction follows.

## Static-Shape Constraint

- gint generates per-shape bytecode: the broadcast plan, tile mode, batch strides, and `grid_dim` are all baked at compile time. SymInt-shaped FakeTensors can't be packed into `ProgramTensorInfo` (the executor's `HTensorInfo` struct needs concrete ints) and would crash at runtime when allocating output tensors.
- **Recommended entry point: `gint.conductor.compile`** — drop-in for `torch.compile` that wraps the call site with `torch._dynamo.config.patch(automatic_dynamic_shapes=False, assume_static_by_default=True)`. The patch is active only while the wrapped callable is executing, so dynamo's tracing (which fires inside that call) sees the patched config — no global side effect on other backends in the same process. Backend-internal patching doesn't work because dynamo's trace decision happens *before* it dispatches to our backend; the patch must live around the user-call site.
- After `cache_size_limit` (default 8) per-shape compiles, dynamo logs a warning and falls back to eager. Users with many shape variants can bump `torch._dynamo.config.cache_size_limit` before the first compile.
- Defensive guard inside `compiler_fn`: if SymInt-shaped tensors still slip through (e.g. user passed `dynamic=True` to `torch.compile`, or used `torch.compile(backend="gint")` directly without the wrapper), raise a `RuntimeError` pointing at the remediation (use `gint.conductor.compile`, pass `dynamic=False`, or `mark_static`). Better to fail loudly than silently emit a kernel for the wrong shape.

```python
from gint.conductor import compile as gint_compile

@gint_compile
def fn(a, b):
    return a + b

# or with options
@gint_compile(options={"cuda_graphs": False})
def fn2(a, b):
    return torch.bmm(a, b) + 1.0
```

## Backend Registration

- `import gint.conductor` auto-registers two backends (only runs when torch is importable, since `gint.conductor` itself depends on torch):
  - `"gint"` — default, `cuda_graphs=True`
  - `"gint-no-cuda-graph"` — legacy alias for `cuda_graphs=False`
- `gint/__init__.py` already imports `gint.conductor` under a try/except, so `import gint` is sufficient on torch-enabled installs; on torch-less installs the conductor module silently isn't loaded and no backends are registered.
- Options are passed via `torch.compile`'s ``options`` dict: ``options={"cuda_graphs": False, "num_warmup_iters": 5}``. The backend function receives these via `**kwargs` from TorchDynamo's `_TorchCompileWrapper`.
- `gint_backend(*, cuda_graphs=True, num_warmup_iters=1)` returns a backend callable with baked-in defaults, which can still be overridden by ``options`` at compile time.
- `register_backend(name, cuda_graphs, num_warmup_iters=1)` is still public for users who want a custom name; it swallows re-registration errors with a printed warning so re-imports are safe. `num_warmup_iters` is forwarded to `make_graphed_callables` — default 1 is enough to populate gint's per-shape device buffer cache before capture; don't pass 0 (allocations would land inside the captured region).
- **AOT autograd**: backend uses `aot_module_simplified` (Inductor's lighter wrapper, not `aot_module`) and passes `inference_compiler=compiler_fn` so the no-grad / `inference_mode()` path skips joint-graph tracing entirely. Saves ~1.5–3 ms per cold compile vs the fuller `aot_module` path. Backward is not supported; passing the same compiler for `fw_compiler` and `inference_compiler` is intentional.

## CUDA Graphs

- The `"gint"` backend wraps the AOT-compiled callable with `torch.cuda.make_graphed_callables`, so subsequent calls replay a CUDA graph instead of paying per-call launch overhead. Use ``options={"cuda_graphs": False}`` to opt out.
- `mode="reduce-overhead"` is **inductor-only** and is silently ignored when `backend="gint"`; pass ``options={"cuda_graphs": True}`` for the gint equivalent.
- Implemented entirely via PyTorch's `make_graphed_callables` — no executor changes. The existing `cuStreamIsCapturing` branch in `CudaExecutor.execute` (`gint/host/cuda/executor_impl.py:82-85`) already cooperates with capture: the pinned `HTensorInfo` host buffer is updated each call (writing fresh tensor base pointers), and the captured H2D memcpy node copies it into the cached device `dinfo` on every replay. The bytecode and tensor-info device buffers are cached per-shape in `program._cu[pcp]` so addresses are stable.
- `cuda_graphs` and `num_warmup_iters` are configured via ``torch.compile``'s ``options`` dict or `gint_backend()`.
- Microbench (relu(x)+y*2.0, N=1024 on a single GPU): ~3× speedup from launch-overhead amortization.
- Not wrapped: `execute_indirect` (per-warp pointer tables are reallocated per call). The conductor's pointwise/reduction path uses `execute`, so this isn't on the hot path. HIP analogue not yet plumbed.

## Broadcasting

- Native NumPy-style broadcasting across per-point operations (e.g., `(32, 128) + (128,)` for bias addition).
- **No kernel changes required** — broadcasting is implemented via `ProgramTensorInfo` stride tricks: `stride=0` for broadcast dimensions causes `offset * 0 = 0`, repeating data without duplication.
- **Key invariant**: All `ProgramTensorInfo` fields that affect grid index decomposition must be identical across tensors in a subgraph (`block_grid_dims`, `block_grid_steps`, `batch_shape`, `block_shape_stride_1[0]`). Only strides differ per tensor: `block_shape_stride_1[1]` and `batch_strides` are 0 for broadcast dims.
- **Broadcast plan** (`_compute_broadcast_plan`):
  1. Merge consecutive innermost dims where ALL tensors match output (non-broadcast) into the block.
  2. Remaining outer dims become batch dims (up to 4, otherwise infeasible).
  3. Per tensor: `block_stride` = 0 if broadcast in inner dims, 1 otherwise; `batch_strides[d]` = 0 for broadcast dims, C-contiguous stride otherwise.
- **Partitioner integration**: shape equality check replaced with `_broadcast_shapes()` compatibility; broadcast plan feasibility is verified before accepting a node.
- Example: `a(32, 128) + b(128,)` → block=128, batch=[32], grid=32 warps. `b` gets `batch_strides=[0]` so each row reuses the same 128 elements.

## Pointwise Tile Dispatch (small inner block)

When `_compute_broadcast_plan` leaves `block_size < 128` (e.g. `relu(M, 1) + n*L(M, 3)` where the inner-dim merge stops at the 1≠3 mismatch), the existing 1d codegen would use only `block_size/128` of each warp's lanes. `_select_pointwise_tiling(block_size)` lifts the same B*R=128 dispatch as reductions onto the pointwise path:
- `block_size >= 128` → 1d (existing flat path; coalesced thread-stride 1)
- `16 <= block_size < 128` → 2dt (B=4 batches × ≤32 inner per warp; thread-stride 1 stays coalesced)
- `block_size < 16` → 2dw (B=32 batches × ≤4 inner per warp; thread-stride is the small inner dim K, ~K cache lines per width-lane amortized over 32× fewer warps)

`StackCodegen` is parameterized with `load_fn` / `store_fn` so the same code emits `fldg_1d`/`2dt`/`2dw` and `fstg_*` per the chosen tile. Pointwise ops are width-lane independent so the codegen logic itself is mode-agnostic — only the load/store frontend differs.

Per-tensor TensorInfo per mode: 2dt places the inner block in shape_1 (thread axis) and the innermost merged batch dim in shape_2 (width axis); 2dw swaps them. Inner chunks (`ceil(block_size / R)`) and batch tiles (`ceil(M / B)`) live in `block_grid_dims`, so partial last tiles get clamped automatically by the kernel's `b_shape - b_idx*step` masking.

Outer batch dims (when len(batch_dims) > 1) stay in `batch_shape`; the innermost is pulled into block_grid. Current implementation only enables the new tile when `len(batch_dims) >= 1`.

Geometry-smith case: `(M, 1)` and `(M, 3)` had `block_size=3, batch_dims=[M]` → 2dw with `grid=ceil(M/32)`. Wall time dropped 0.70 ms → 0.16 ms at M=1M (matches eager within 1.5×).

## Commutative-Op Swap Elision

- `OpDescriptor.commutative` flags arity-2 ops whose result is invariant under arg order: `add.Tensor`, `mul.Tensor`, `eq.Tensor`, `ne.Tensor`, `maximum.default`, `minimum.default`.
- `StackCodegen.try_skip_commutative_args(node, op_desc)` runs before normal arg processing in `_compile_subgraph`. When both args are already on the virtual stack at depths 0 and 1 (in either order, last-use, neither in tensor_map / regs), it consumes them in place and skips the Swap that the depth-1 path would emit. The op then runs on whichever order the stack has — same result.
- Why it matters: e.g. geometry_smith's final `mul_6 = mul.Tensor(div_1, div_3)`. After computing `div_3` from FRDiv, the stack has `[..., div_1, div_3]`. The declared arg order `[0, 1]` would force pushing `div_1` first (Swap), then `div_3` (Swap again — the second Swap was just to undo the first). With the flag, both Swaps disappear. K#2 dropped from 36 → 34 instructions.
- Steady-state wall time barely moves (kernels are bandwidth-bound, not insn-bound), but the bytecode is tighter and the analyzer's `max_stack` may now allow `s7` for borderline subgraphs.

## CSE Pre-Pass

- `GintCompiler.compile` runs `torch.fx.passes.dialect.common.cse_pass.CSEPass` on the AOT graph before partitioning. The pass is hash-based local value numbering on `(target, hash(args, kwargs))` with a banlist (random / in-place ops) supplied via `get_CSE_banned_ops()`. Returns a fresh `GraphModule` with node metadata preserved.
- Why: AOT decomposition often emits duplicates that FX's own tracing didn't catch. Geometry_smith's two `schlick_ggx(_, roughness)` calls each materialize their own `r*r`, `/2`, `1-k` — CSE collapses the duplicates so the partitioner / register planner sees a single canonical copy. Cuts the post-reduction tail from 14 nodes (21 insns) to 11 nodes (18 insns).
- Functional dialect only — gint never emits stateful or in-place ops, so the standard banlist suffices.

## Register-Spill Codegen (`_plan_spills`)

- The l12 kernel variant has `POOL_SIZE=12` slots shared between stack (grows up from `pool[0]`) and 8 virtual registers (`reg N = pool[POOL_SIZE-1-N]`). The conductor uses up to 11 of those (`stack + active_regs ≤ 11`) to leave one slot of headroom for transient `dup`/`swap` shuffling.
- `_plan_spills(nodes, ext_io)` walks the schedule and assigns a virtual register to two classes of node:
  1. **Multi-use intermediates** (`>1 distinct downstream user inside the subgraph`, not in `ext_io`).
  2. **Buried single-use nodes** discovered by simulating the abstract stack in the codegen's actual `arg_order` and watching for any internal arg that lands at depth ≥ 2 by the time its `handle_operand` fires. Codegen has no rotN — only `swap` rescues depth 1 — so a buried single-use value would otherwise raise the "buried at depth ≥ 2" runtime error in `StackCodegen.handle_operand`. Burial detection iterates: each round adds the newly-buried args to the spill set and re-simulates, converging in O(nodes) rounds since each round adds ≥ 1 node or terminates.
- Register indices are reused across non-overlapping live ranges via simple liveness (free regs at last-use, allocate at result production). Returns `(node_to_reg, overflow, feasible)` — overflow is the set of nodes that couldn't fit in any of the 8 registers and falls back to a global-tensor slot.
- `GraphPartitioner._stack_fits` is a thin wrapper over `_plan_spills(...).feasible`. Strictly more permissive than the old depth-only check because spilling a long-lived multi-user value to a register removes it from the abstract stack.
- `GraphPartitioner._required_globals = ext_io ∪ overflow` — multi-user intermediates that fit in registers don't get a tensor slot, freeing both `max_tensors` budget and global memory traffic.
- `StackCodegen` accepts the spill plan via `node_to_reg`. `emit_op` emits `FStoreReg<N>` after computing a spilled node (and pops the value from the virtual stack); `handle_operand` emits `FLoadReg<N>` whenever an arg is needed and isn't on the virtual stack at depth 0/1. The generic-arg, depth-1, and depth-≥2 branches all prefer a register reload over a global reload.
- The host's `select_variant` automatically picks `l12` when the bytecode references registers (`max_reg_idx ≥ 4`); subgraphs with no spills stay on `s7`.
- Geometry_smith post-reduction tail: 14-node pointwise chain (two `schlick_ggx` calls + final multiply) used to split into a 10-node + 4-node pair (stack overflow at op 11). With register spills, all 14 ops fuse into a single 21-instruction kernel using regs 0 and 1 for `relu_1` and `k₁`. Wall time at M=1M: 115 µs → 77 µs (eager 102 µs).

## Metadata Ops

- The conductor supports shape/stride-only ops (`view`, `unsqueeze`, `squeeze`, `expand`, `permute`, `transpose`, `t`, `slice`) as **identity on the stack** — no bytecode emitted, value passes through unchanged.
- Registered as `OpKind.METADATA` in `op_registry.py` with `arity=1`, `arg_order=[0]` (only the tensor arg is pushed; shape/dim args are read from the FX node, not the stack).
- Enables fusion through metadata ops: e.g., `relu(x).unsqueeze(0) + 1.0` compiles to a single kernel instead of relu → eager unsqueeze → add (two kernels).
- Shape changes are handled by the broadcast plan's per-tensor `ProgramTensorInfo` strides — unsqueeze adds a size-1 dim that broadcasts naturally, expand sets stride=0 for expanded dims.
- **Slice support** (`aten.slice.Tensor`): slice + pointwise fuses into a single kernel (e.g., `x[:, :64].relu()`). The slice input global uses effective shapes (sliced size). Non-zero start offsets are handled via `narrow()` at runtime (`input_adjustments` in `GintCompiledSubgraph`). Stepped slices (step != 1) are not fused.
- **`_get_strides`**: reads actual tensor strides from FX metadata (`node.meta['val'].stride()`). Falls back to C-contiguous computation if unavailable. Strides are threaded through `_compute_broadcast_plan` for correct `block_stride` and `batch_strides` on non-contiguous tensors.
- **Partitioner safety**: when a metadata op's global input shape is not broadcast-compatible with the output shape (e.g., `view(1024) → (32, 32)`), the node is skipped and falls back to eager execution. Slice ops are forced to start a new subgraph (they shrink a dimension which isn't broadcast-compatible).
- **Current limitation**: non-contiguous input strides (e.g. transpose) are correctly read and used for input tensors, but the output tensor is still created as C-contiguous. The output stride doesn't match the input stride for transposed views, so `x.t().relu()` still falls back to eager.

## Reduction Support (sum, mean, prod, amax, amin)

- Registered as `OpKind.REDUCTION` in `op_registry.py` for `aten.sum.dim_IntList`, `aten.mean.dim`, `aten.prod.dim_int`, `aten.amax.default`, `aten.amin.default`.
- **Constraint**: innermost-dim only, single reduction dim. Non-innermost reductions fall back to eager.
- **Tile dispatch (`_select_reduction_tiling`)**: each warp packs `B * R == 128` lanes covering `B` batches × `R` reduction elements. Choice driven by innermost dim N:
  - `N >= 128` → (B=1, R=128, mode `1d`); threads+width flatten over reduction; final = warp_allreduce + width-combine.
  - `16 <= N < 128` → (B=4, R=32, mode `2dt`); threads scan reduction, width-lanes carry 4 batches; final = warp_allreduce only.
  - `N < 16` → (B=32, R=4, mode `2dw`); threads carry 32 batches, width-lanes scan reduction; final = width-combine only.
  - Both `_compile_reduction_subgraph` and `_compile_fused_reduction_subgraph` use this dispatch. The small-N (mode != '1d') fused path supports all three phase combinations — pre_prefix, scalar_prefix, and broadcast_suffix. Mode '1d' (large N) stays on the legacy fused codegen because its layout (batches in `batch_shape`, not `block_grid`) is structurally different and well-tested by RMSNorm.
- **`combine_fn` / `warp_reduce_fn` / `post_reduce_fn`**: descriptor splits the reduction's emit into pieces so the compiler can call only the relevant ones per tile. `combine_fn` is the pairwise multi-chunk accumulator (fadd for sum/mean, fmul for prod, `dup2; fgt/flt; fselect` pair for amax/amin). `warp_reduce_fn` is the 1-instr `warp_allreduce_*`. `post_reduce_fn(node)` is the optional final fix-up (mean's `* 1/N`).
- **OOB padding caveat**: kernel pads OOB lanes with `0.0`. Sum/mean tolerate this; prod (× 0), amax (max-with-0 clamps negatives), amin (min-with-0 clamps positives) do NOT. Those three are gated by `_check_reduction_feasible_clean_chunks`, which now requires `N % R == 0` for the chosen tile (was `N % 128 == 0`). E.g. N=64 with tile R=32 is now accepted; previously rejected.
- **No kernel changes required** — width-lane combining is composed from existing instructions:
  ```
  warp_allreduce_fsum     ; [p0, p1, p2, p3] per thread
  dup; fperm_w(2,3,0,1); fadd   ; [p0+p2, p1+p3, ...]
  dup; fperm_w(1,0,3,2); fadd   ; [total, total, total, total]
  ```
  Factored as `_emit_width_combine(combine_fn)` and reused per-tile by `_compile_reduction_subgraph`. Mean appends `fmulimm(1/N)` via `post_reduce_fn`.
- **Batch-dim merge (`_merge_dims`)**: greedy incremental merge of consecutive batch dims that are C-contiguous in *every* tensor of the subgraph. Unlike `_compute_broadcast_plan`'s all-or-nothing merge, this stops at non-contiguous boundaries without collapsing back to length-1, so partial merges still help.
- **Per-warp tile clamping**: the innermost merged batch dim is placed in `block_grid_dim_2` (modes 1d/2dt) or `block_grid_dim_1` (mode 2dw) with step=B, so the kernel's `b_shape - b_idx*step` clamping handles partial last tiles when `M % B != 0`. Outer batch dims stay in `batch_shape`.
- **Multi-chunk loads**: for reduction dim > R, loads are unrolled at Python codegen time: `fldg_1d/2dt/2dw(0, slot)`, `(R, slot)`, `(2R, slot)`, … each followed by `combine_fn`. Practical for N up to ~16K.
- **Partitioner**: reduction nodes are isolated as single-node subgraphs (flush any current pointwise subgraph).

## Fused Reduction+Broadcast+Pointwise

- When a `keepdim=True` reduction is followed by a pointwise subgraph, the conductor fuses them post-partitioning into a single subgraph.
- **Generalized multi-op support**: the post-reduction pointwise chain is split into a **scalar prefix** (ops consuming only the reduction scalar, e.g. `+eps`, `rsqrt`) and a **broadcast suffix** (ops consuming external globals, e.g. `*x`, `*w`). The scalar prefix executes once; the broadcast suffix runs per-chunk.
- **Pre-reduction fusion**: A pointwise subgraph immediately preceding a reduction (e.g. `x*x` before `mean`) is absorbed into the Phase 1 accumulation loop — no intermediate global memory round-trip. `_split_pre_prefix(pre_schedule, reduction_input)` further splits pre_prefix into:
  - **per_chunk**: ancestors of the reduction input (e.g. `mul_1 = n*L` feeding `sum`) — emitted inside Phase 1's chunk loop, external inputs must be reduction-dim-shaped.
  - **per_batch**: ops independent of the reduction chain that operate on `[..., 1]`-shaped per-batch values (e.g. `relu(sum_1)` running alongside an unrelated `n*L → sum_2` reduction) — emitted ONCE in Phase 0 before the chunk loop, external inputs broadcast across chunk lanes (stride 0 in inner axis), external outputs stored once via `fstg_2dt` writing only the size-1 axis. Per-batch external inputs and outputs are added to `scalar_globals` so their `ProgramTensorInfo` gets the size-1 inner stride.
- **Four-phase bytecode** (mirrors the handwritten rmsnorm kernel in `tests/test_rmsnorm.py`):
  - Phase 0: per_batch_pre — load per-batch globals (`block_load_fn(0, slot)` with stride-0 inner), apply ops, store outputs via `scalar_store_fn`.
  - Phase 1: For each chunk, load + apply per_chunk_pre → accumulate, then warp_allreduce + width-lane combine → scalar on stack.
  - Phase 1b: emit scalar prefix ops on the scalar.
  - Phase 2: For each chunk, `dup` scalar, load chunked globals, emit broadcast suffix ops, store output.
- **Two codegen paths**:
  - `_compile_fused_reduction_new_tile` for small N (modes 2dt and 2dw): tile-aware Phase 1 (chunked `fldg_2dt/2dw` + pre_prefix ops + accumulate), Phase 1b (scalar prefix on per-thread/per-width-lane scalar), and Phase 2 (per-chunk broadcast suffix — `dup` scalar, load external block-shaped globals via `block_load_fn(k*R, slot)`, run pointwise ops, store via `block_store_fn(k*R, slot)`). Per-tensor TensorInfo distinguishes "scalar" globals (single-element-per-batch — only the reduction-only output when broadcast_suffix is empty) from "block" globals (full reduction-dim tile, shape ending in N — every other global). Per-tensor batch strides pad with 0s on the left for tensors with fewer batch dims, and use 0 within-shape for size-1 broadcast dims; `_merge_dims` runs over all tensors jointly so partial merges are honored.
  - `_compile_fused_reduction_subgraph` (legacy 1d) for mode '1d' only (N ≥ 128). Structurally distinct (batches in `batch_shape` vs `block_grid_dim_2`) and production-tested by RMSNorm.
- **Geometry_smith fused-reduction** at N=3 (mode 2dw, pre_prefix split into per_batch=[relu]+per_chunk=[mul_1], no suffix): collapses the `relu(sum_1) + (n*L).sum(-1)` middle pair into one kernel, dropping the outer per-iter from 4 kernels (75 µs) to 3 kernels (63 µs) at M=1M. **Small-N RMSNorm-shape** at N=8 or N=64 (mode 2dw / 2dt, pre_prefix=[mul] + scalar_prefix=[+eps, rsqrt] + broadcast_suffix=[*x, *w]): single fused kernel covered by `tests/test_conductor_real_workloads.py::test_rms_norm_small_N_2dw / _2dt`.
- **Fusion detection**: `_can_fuse_with_reduction` (post-reduction) and `_can_fuse_pre_reduction` (pre-reduction) validate that all involved ops are ELEMENTWISE, that the reduction result has no external users, and that all external globals share the reduction dimension.
- **Deferred compilation**: pointwise subgraphs preceding reductions are buffered rather than compiled immediately, avoiding wasted work when they are absorbed by pre-reduction fusion. The buffer is a *list* of pending pointwise schedules: when the partitioner splits a per-batch op (e.g. `relu(scalar_in)`) and a per-chunk op (e.g. `n*L`) into separate subgraphs (their outputs disagree in shape — see *Output-Driven Iteration Shape*), the merger first tries `_can_fuse_pre_reduction` on the **concatenation** of every pending schedule. If that succeeds, all per-batch + per-chunk ops absorb into one fused-reduction kernel. If it fails, the merger falls back to fusing only the most recent schedule (compiling earlier ones as standalone kernels), then to compiling all pending separately.
- **Stack depth**: Phase 1 peaks at ~2 extra slots (width-lane combine). Phase 2 needs scalar + 1 for input + op overhead. Total ~4-5, well within MAX_STACK=8.
- Examples: `x - mean(x)` (single binary op, existing pattern). `x * rsqrt(mean(x*x) + eps) * w` (full RMSNorm, 6 nodes fused into 1 kernel).

## Small-Matrix BMM / Inverse Rewrite

- `gint/conductor/special_ops.py` runs a pre-partition FX rewrite that targets two patterns from `torch.compile`'s AOT graph:
  - `aten.bmm.default(a, b)` where both args have trailing dims `(N, N)` with N ≤ 4 (and N matches across a / b).
  - `operator.getitem(aten.linalg_inv_ex.default(a), 0)` (the `linalg.inv` decomposition) where `a` has trailing dims `(N, N)` with N ≤ 4. The dead `linalg_inv_ex` node is erased after the rewrite.
- Both patterns are replaced with calls to Python wrappers in `gint/host/matrix.py` (`gint_bmm`, `gint_inv`). The wrappers' targets aren't in `OP_REGISTRY`, so the partitioner skips them — they run via `_run_eager`, which calls our wrapper, which dispatches to a `SugarProgram`. This avoids touching the partitioner / pointwise codegen logic.
- Kernel sharing: only the 4×4 BMM and INV kernels exist (mirrors of `tests/test_bmm4x4.py` / `tests/test_inv4x4.py`). Smaller N is handled by padding before launch:
  - BMM: pad A and B with zeros into the bottom rows/columns of a 4×4 buffer. Zero rows/columns contribute nothing to A·B, so the top-left N×N of the 4×4 product is `a @ b`. Sliced back at the end.
  - INV: pad with the identity block — `diag(A, I_{4-N})`. The padded matrix is block-diagonal, so its inverse is `diag(inv(A), I_{4-N})`; the top-left N×N is `inv(A)`. Sliced back at the end.
  - N == 1: skipped — `gint_bmm` returns `a * b`, `gint_inv` returns `a.reciprocal()`. No kernel launch.
- Surrounding pointwise ops still fuse through gint codegen normally — only the bmm/inv node itself is on the eager-fallback path. Verified by `tests/test_conductor_bmm_inv.py::test_subgraphs_post_rewrite`: the trailing `+ 1.0` after a bmm becomes a separate single-node pointwise gint subgraph.
- Padding allocates fresh tensors on each call (PyTorch's caching allocator handles this under cuda-graph capture). For N=2 this is 4× compute waste; acceptable given the launch-overhead amortization (small bmm/inv are launch-bound).
- N > 4 is not rewritten and falls through to torch.bmm / torch.linalg.inv eager. Non-square matrices and non-matching `(N, M, K)` BMM also skip the rewrite.

## Reduction Future Work

- **Unify mode '1d' onto the new tile dispatch** — currently the legacy `_compile_fused_reduction_subgraph` handles mode '1d' (N ≥ 128) with batches in `batch_shape` and per-tensor block_size, while the new path uses `block_grid` and joint-merged batches. Migrating mode '1d' would remove the legacy codegen and unify the three modes; needs careful testing against RMSNorm.
- **Dedicated `WarpFullReduceSum` instruction** — single opcode replacing the 7-instruction width-lane combine sequence.
- **Kernel loop instruction** — for reduction dims > ~16K without bytecode bloat.
- **Non-innermost reductions** — would require transposing the tensor or a different grid decomposition.
- **Multi-reduction patterns** — e.g., layer norm (mean + variance in one pass).
