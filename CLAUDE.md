# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Gint** is an experimental project implementing a **completely GPU device-side interpreter architecture** for kernel programming with cross-OS and cross-device-vendor support. The project is written with Python + LLVM IR, enabling efficient bytecode execution directly on the GPU.

The codebase is split into:
- **Host side**: Python + CUDA/HIP bindings for device memory management, kernel loading, and bytecode dispatch
- **Kernel side**: LLVM IR interpreter executing stack-based bytecode on GPU (platform-agnostic)
- **Conductor**: PyTorch `torch.compile` backend integration

Supported GPU backends:
- **NVIDIA (CUDA)**: Turing through Blackwell via `cuda-bindings` + fatbin
- **AMD (HIP)**: RDNA3/4 (wave32) via `hip-python` + per-target HSACO

## Architecture

### Stack-Based Interpreter (Device-Side)
- Located in `gint/kernel/interpreter/` - written in LLVM IR via `llvmlite`
- **Micro-architecture**:
  - `REG_WIDTH = 4`: Processes 4 elements simultaneously (ILP - Instruction Level Parallelism)
  - Pool/stack/register sizes are per-variant (see Kernel Variants below). Default `l12` variant: `POOL_SIZE=12, NUM_REGS=8, MAX_STACK=8`. Stack grows upward from index 0, virtual registers occupy the top `NUM_REGS` slots counting downward (`reg n = pool[POOL_SIZE-1-n]`).
  - Dispatch: Large switch-case in LLVM IR with weights for optimization
  - Uses PHI nodes for state persistence across instruction dispatches (one dispatch block per stack depth 0..MAX_STACK)
- **Instruction set**: Defined in `gint/kernel/interpreter/instructions/`

### Kernel Variants
- The fatbin/HSACO ships **two kernel variants** under symbols `geval_s7` and `geval_l12`. Both implement the full interpreter; they differ only in pool/stack/reg sizes:
  - `s7`: `POOL_SIZE=7, NUM_REGS=4, MAX_STACK=7` — fits all pointwise/streaming workloads (max measured: stack=6, regs=0). Smaller register pressure on the GPU → better occupancy.
  - `l12`: `POOL_SIZE=12, NUM_REGS=8, MAX_STACK=8` — required by register-heavy kernels (e.g. `inv4x4` uses all 8 regs).
- **Variant table**: `gint.kernel.interpreter.main.VARIANTS` maps name → `(pool_size, num_regs, max_stack)`. `variant_kernel_name(name)` formats the kernel symbol (`geval_<name>`).
- **IR generation**: `build_interpreter_main_nvptx()` / `build_interpreter_main_amdgcn()` emit ONE module containing every variant's kernel. The `dynamic_smem` shared global is created once and shared across kernels. AMDGCN's `emit()` post-processor attaches the `#0` attribute group to every kernel symbol.
- **Per-variant opcode filtering**: `FLoadRegN`/`FStoreRegN` opcodes for `N >= num_regs` are dropped from the dispatch table of smaller variants (otherwise they would alias into normal stack slots via `pool[pool_size-1-N]`). The `s7` variant only accepts opcodes 87–90 (load reg 0–3) and 95–98 (store reg 0–3).
- **Host-side selection** (`gint/host/executor.py:select_variant`): runs `analyze_bytecode` on each compiled program, picks the smallest variant whose `(num_regs, max_stack)` covers the program. Accepts either raw bytecode or a `ProgramData` (preferred — reuses the lazily-cached `pd.stats`, since `analyze_bytecode` walks the whole stream and is the dominant cost on the cache-miss path). Variant decision is cached alongside the per-program device buffers in `program._cu` / `program._hip`.
- **`execute_indirect`**: picks `max(variant) over all programs in the batch` (one launch per call, not bucketed). Mixed batches are rare; bucketing would multiply launch overhead.
- **Build/deploy**: `./generate.sh` and `./generate_amdgcn.sh` bundle BOTH variants into one fatbin/HSACO. Re-run after pulling these changes — the old single-`geval` artifacts are incompatible with the new selection code.

### Virtual Register File
- 8 virtual registers (`reg 0`–`reg 7`) backed by the top 8 slots of the unified pool
- `FLoadRegN` / `FStoreRegN` (N=0..7): Specialized per-register instructions with **zero select/compare overhead** — each directly aliases `pool[pool_size-1-N]`
- Generic `FLoadReg` / `FStoreReg` with a runtime operand were replaced by 16 specialized opcodes (87–102) to eliminate register pressure from select chains
- Frontend: `fload_reg(n)` / `fstore_reg(n)` dispatch to the correct specialized class via `LOAD_REGS[n]` / `STORE_REGS[n]`

### Width-Lane Permutation and Shuffle

#### `binary_cond_tree` utility (`move.py`)
- Module-level function (not a method) used by `DupBroadcastW`, `FPermW`, and `FShuf2`
- Builds a binary selection tree over a list of `ir.Value`s, selecting one based on a runtime i32 index

#### `FPermW` (opcode 104)
- Permutes the 4 width-lanes of the top-of-stack vector in-place
- Operand is an i32 encoded as **i8x4 little-endian**: bits 7..0 = source lane for output 0, bits 15..8 = source lane for output 1, etc.
- Frontend: `fperm_w(i0, i1, i2, i3)` encodes the four indices and emits opcode 104
- **Broadcast idiom**: `fperm_w(i, i, i, i)` broadcasts lane i in-place — preferred over `dup_broadcast_w(i); swap(); pop()` (1 instruction vs 3)

#### `FShuf2` (opcode 105)
- Two-source shuffle: `VecShuffle(vec1, vec2, x, y, z, w)` → `(vec1[x], vec1[y], vec2[z], vec2[w])`
- vec2 is on top of stack, vec1 is below; pops both, pushes shuffled result
- Same i8x4 little-endian operand encoding as `FPermW`; indices x,y select from vec1, z,w from vec2
- Frontend: `fshuf2(x, y, z, w)` encodes indices and emits opcode 105

### Indirect Load/Store (Scatter/Gather)
- `LoadGlobal1DF32Indirect` / `StoreGlobal1DF32Indirect`: 1D indirect (gather/scatter) using stack-provided indices
- Stack protocol: push values (store only), then push indices, then call the indirect instruction
- Indices are f32-bitcast-to-i32 on the stack; `advance_ptr_cond_state` bitcasts back to i32 for addressing
- **Width-lane iteration contract**: The base `emit` loop in `_LoadStoreGlobalBase` calls `advance_ptr_cond_state(w + 1, ...)` after processing lane `w`, because `advance` receives the lane index it is *setting up* (not the one just completed). `init_ptr_cond_state` bootstraps lane 0 by calling `advance(w=0, ...)`

### Indirect Dispatch Mode
- The kernel accepts a `flag: i32` as its 5th argument
- **Direct mode** (`flag <= 0`): All warps share the same `arg(0)` (bytecode pointer) and `arg(1)` (TensorInfo pointer) — the original behavior
- **Indirect mode** (`flag > 0`): `arg(0)` and `arg(1)` are treated as **pointer tables** indexed by `logical_program_idx()`, so each warp loads its own program and tensor info via `ptr = ptrs[idx]`
- Resolution uses `if_else` + phi to avoid speculative loads on invalid pointers in direct mode
- The early-exit bounds check (`logical_program_idx() >= arg(3)`) applies in both modes — in indirect mode, `arg(3)` is the total number of programs in the table
- After pointer resolution, the rest of the interpreter (dispatch loop, tensor info loading, instruction execution) is unchanged

### Host-Side Executor
- `gint/host/executor.py`: Core program execution interface + `get_executor()` auto-detection
- `gint/host/cuda/executor_impl.py`: NVIDIA-specific implementation (CudaExecutor)
  - Loads kernel from `gint/host/cuda/gint.fatbin.xz` (not tracked in git — must be generated via `./generate.sh`)
- `gint/host/hip/executor_impl.py`: AMD-specific implementation (HipExecutor)
  - Loads kernel from `gint/host/hip/gint.fatbin.xz` (not tracked in git — must be generated via `./generate_amdgcn.sh`)
- **Backend selection**: `GINT_BACKEND` env var (`"cuda"` or `"hip"`) for explicit override; default tries CUDA first, falls back to HIP
- `gint/host/utils.py`: Shared utilities (`fill_tensor_info`, `cdiv`) used by both executors
- **`execute_indirect(programs, args_list, indices)`**: Launches multiple different programs in a single kernel call
  - `programs`: list of `BaseExecutableProgram` instances
  - `args_list`: list of tensor arg tuples, one per program (must match `programs` length)
  - `indices`: int array of size `grid_dim`, mapping each warp to a program index
  - Builds per-program device code and tensor info, then assembles per-warp pointer tables for the kernel's indirect mode (`flag=1`)
  - Tensor info grid dimensions interact with `logical_program_idx()` via `urem`, so contiguous warp assignments per program work naturally

### Frontend API
- `gint/host/frontend.py`: Pythonic bytecode generation
  - `@bytecode` decorator: Intercepts Python calls to record instruction sequences
  - `ProgramTensorInfo`: Encapsulates tensor metadata (strides, shapes, element sizes)
  - Each instruction emits a 2-word pair `[opcode, operand]`; operand is 0 for instructions that don't use it
- `gint/host/sugar.py`: Higher-level convenience functions
- `gint/host/analyzer.py`: Static analyzer over recorded bytecode — `analyze_bytecode(bc)` returns a `BytecodeStats` (peak stack depth, registers used, min pool size). Hand-maintained `_EFFECTS` table covers every opcode in `INSNS`. Torch-free; useful for kernel-variant selection and partitioner heuristics. Re-exported as `gint.analyze_bytecode`

### Torch.Compile Integration (Conductor)
- `gint/conductor/backend.py`: Backend registration and entry point
- `gint/conductor/compiler.py`: FX graph → bytecode conversion, graph partitioning, and broadcasting
- `gint/conductor/debug.py`: `inspect_subgraphs(fn, *args)` runs `fn` through the gint backend and returns one `SubgraphInfo` per compiled subgraph (kind, FX nodes, bytecode, output shape, grid dim). Pair with `print_subgraphs` to dump the bytecode of each subgraph — the right tool when a torch.compile path looks slower than expected or you want to verify partitioning / fusion behavior
- Supported op surface (full list in `op_registry.py`; unsupported ops fall back to eager):
  - Arithmetic: add, sub, mul, div, remainder, neg, abs, square, reciprocal, pow, atan2, rsub.Scalar
    - **Scalar fold**: `add/sub/mul/div` with a Python-scalar second arg lower to a single immediate insn (`faddimm` / `fmulimm`) instead of `LoadImm + fadd/fmul`. Driven by `OpDescriptor.arg_order` accepting a callable so the descriptor decides at emit time whether to push the scalar or skip it. `emit_op` takes a `num_pushed` count instead of relying on static `arity` so the codegen pop matches actual pushes
  - Comparisons + ternary: gt/lt/ge/le/eq/ne (Tensor + Scalar), `where`
  - Transcendentals: sqrt, rsqrt, exp, exp2, expm1, log, log2, log10, log1p, sin/cos/tan, asin/acos/atan, sinh/cosh, atanh/asinh/acosh, erf
  - Activations: relu, gelu, silu, leaky_relu, tanh, sigmoid, hardtanh, relu6, hardsigmoid, hardswish, softplus (beta=1), mish, elu, selu, threshold, hardshrink
  - Pairwise + clamp: minimum, maximum, clamp/clamp_min/clamp_max (scalar + Tensor)
  - Composite: addcmul, addcdiv, lerp.Scalar
  - Metadata: view, unsqueeze, squeeze, expand, permute, transpose, t, slice
  - Reductions (innermost dim only): sum, mean, prod, amax, amin
- Partitioner constraints per subgraph: max 8 global tensor slots, max stack depth 8, broadcast-compatible shapes

#### Backend Registration
- `import gint.conductor` auto-registers two backends (only runs when torch is importable, since `gint.conductor` itself depends on torch):
  - `"gint"` — default, `cuda_graphs=True`
  - `"gint-no-cuda-graph"` — legacy alias for `cuda_graphs=False`
- `gint/__init__.py` already imports `gint.conductor` under a try/except, so `import gint` is sufficient on torch-enabled installs; on torch-less installs the conductor module silently isn't loaded and no backends are registered
- Options are passed via `torch.compile`'s ``options`` dict: ``options={"cuda_graphs": False, "num_warmup_iters": 5}``. The backend function receives these via `**kwargs` from TorchDynamo's `_TorchCompileWrapper`.
- `gint_backend(*, cuda_graphs=True, num_warmup_iters=1)` returns a backend callable with baked-in defaults, which can still be overridden by ``options`` at compile time.
- `register_backend(name, cuda_graphs, num_warmup_iters=1)` is still public for users who want a custom name; it swallows re-registration errors with a printed warning so re-imports are safe. `num_warmup_iters` is forwarded to `make_graphed_callables` — default 1 is enough to populate gint's per-shape device buffer cache before capture; don't pass 0 (allocations would land inside the captured region)
- **AOT autograd**: backend uses `aot_module_simplified` (Inductor's lighter wrapper, not `aot_module`) and passes `inference_compiler=compiler_fn` so the no-grad / `inference_mode()` path skips joint-graph tracing entirely. Saves ~1.5–3 ms per cold compile vs the fuller `aot_module` path. Backward is not supported; passing the same compiler for `fw_compiler` and `inference_compiler` is intentional

#### CUDA Graphs
- The `"gint"` backend wraps the AOT-compiled callable with `torch.cuda.make_graphed_callables`, so subsequent calls replay a CUDA graph instead of paying per-call launch overhead. Use ``options={"cuda_graphs": False}`` to opt out
- `mode="reduce-overhead"` is **inductor-only** and is silently ignored when `backend="gint"`; pass ``options={"cuda_graphs": True}`` for the gint equivalent
- Implemented entirely via PyTorch's `make_graphed_callables` — no executor changes. The existing `cuStreamIsCapturing` branch in `CudaExecutor.execute` (`gint/host/cuda/executor_impl.py:82-85`) already cooperates with capture: the pinned `HTensorInfo` host buffer is updated each call (writing fresh tensor base pointers), and the captured H2D memcpy node copies it into the cached device `dinfo` on every replay. The bytecode and tensor-info device buffers are cached per-shape in `program._cu[pcp]` so addresses are stable
- `cuda_graphs` and `num_warmup_iters` are configured via ``torch.compile``'s ``options`` dict or `gint_backend()`
- Microbench (relu(x)+y*2.0, N=1024 on a single GPU): ~3× speedup from launch-overhead amortization
- Not wrapped: `execute_indirect` (per-warp pointer tables are reallocated per call). The conductor's pointwise/reduction path uses `execute`, so this isn't on the hot path. HIP analogue not yet plumbed

#### Broadcasting Support
- The conductor natively supports NumPy-style broadcasting across per-point operations (e.g., `(32, 128) + (128,)` for bias addition)
- **No kernel changes required** — broadcasting is implemented via `ProgramTensorInfo` stride tricks: `stride=0` for broadcast dimensions causes `offset * 0 = 0`, repeating data without duplication
- **Key invariant**: All `ProgramTensorInfo` fields that affect grid index decomposition must be identical across tensors in a subgraph (`block_grid_dims`, `block_grid_steps`, `batch_shape`, `block_shape_stride_1[0]`). Only strides differ per tensor: `block_shape_stride_1[1]` and `batch_strides` are 0 for broadcast dims
- **Broadcast plan** (`_compute_broadcast_plan` in `compiler.py`):
  1. Merge consecutive innermost dims where ALL tensors match output (non-broadcast) into the block
  2. Remaining outer dims become batch dims (up to 4, otherwise infeasible)
  3. Per tensor: `block_stride` = 0 if broadcast in inner dims, 1 otherwise; `batch_strides[d]` = 0 for broadcast dims, C-contiguous stride otherwise
- **Partitioner integration**: Shape equality check replaced with `_broadcast_shapes()` compatibility; broadcast plan feasibility is verified before accepting a node into a subgraph
- Example: `a(32, 128) + b(128,)` → block=128, batch=[32], grid=32 warps. `b` gets `batch_strides=[0]` so each row reuses the same 128 elements

#### Pointwise Tile Dispatch (small inner block)
- When `_compute_broadcast_plan` leaves `block_size < 128` (because the partitioner joined mismatched-shape ops, e.g. `relu(M, 1) + n*L(M, 3)` where the inner-dim merge stops at the 1≠3 mismatch), the existing 1d codegen would use only `block_size/128` of each warp's lanes. `_select_pointwise_tiling(block_size)` lifts the same B*R=128 dispatch as reductions onto the pointwise path:
  - `block_size >= 128` → 1d (existing flat path; coalesced thread-stride 1)
  - `16 <= block_size < 128` → 2dt (B=4 batches × ≤32 inner per warp; thread-stride 1 stays coalesced)
  - `block_size < 16` → 2dw (B=32 batches × ≤4 inner per warp; thread-stride is the small inner dim K, ~K cache lines per width-lane amortized over 32× fewer warps)
- `StackCodegen` is parameterized with `load_fn` / `store_fn` so the same code emits `fldg_1d`/`2dt`/`2dw` and `fstg_*` per the chosen tile. Pointwise ops are width-lane independent so the codegen logic itself is mode-agnostic — only the load/store frontend differs
- Per-tensor TensorInfo per mode: 2dt places the inner block in shape_1 (thread axis) and the innermost merged batch dim in shape_2 (width axis); 2dw swaps them. Inner chunks (`ceil(block_size / R)`) and batch tiles (`ceil(M / B)`) live in `block_grid_dims`, so partial last tiles get clamped automatically by the kernel's `b_shape - b_idx*step` masking
- Outer batch dims (when len(batch_dims) > 1) stay in `batch_shape`; the innermost is pulled into block_grid. The current implementation only enables the new tile when `len(batch_dims) >= 1` (i.e. the merge couldn't fully collapse to a flat block); otherwise stays on 1d
- Geometry-smith case: subgraph mixing `(M, 1)` and `(M, 3)` had `block_size=3, batch_dims=[M]` → 2dw with `grid=ceil(M/32)`. Wall time dropped 0.70 ms → 0.16 ms at M=1M (matches eager within 1.5×)

#### CSE Pre-Pass
- `GintCompiler.compile` runs `torch.fx.passes.dialect.common.cse_pass.CSEPass` on the AOT graph before partitioning. The pass is hash-based local value numbering on `(target, hash(args, kwargs))` with a banlist (random / in-place ops) supplied via `get_CSE_banned_ops()`. It returns a fresh `GraphModule` with node metadata preserved.
- Why: AOT decomposition often emits duplicates that FX's own tracing didn't catch. Geometry_smith's two `schlick_ggx(_, roughness)` calls each materialize their own `r*r`, `/2`, `1-k` — CSE collapses the duplicates so the partitioner / register planner sees a single canonical copy. Cuts the post-reduction tail from 14 nodes (21 insns) to 11 nodes (18 insns).
- Functional dialect only — gint never emits stateful or in-place ops, so the standard banlist suffices.

#### Register-Spill Codegen (`_plan_spills`)
- The l12 kernel variant has `POOL_SIZE=12` slots shared between stack (grows up from `pool[0]`) and 8 virtual registers (`reg N = pool[POOL_SIZE-1-N]`). The conductor uses up to 11 of those (`stack + active_regs ≤ 11`) to leave one slot of headroom for transient `dup`/`swap` shuffling.
- `_plan_spills(nodes, ext_io)` walks the schedule once and assigns a virtual register to every multi-user intermediate (`>1 distinct downstream user inside the subgraph`) that isn't already an external-IO node. Register indices are reused across non-overlapping live ranges via simple liveness (free regs at last-use, allocate at result production). Returns `(node_to_reg, overflow, feasible)` — overflow is the set of multi-user nodes that didn't fit in any of the 8 registers and falls back to a global-tensor slot.
- `GraphPartitioner._stack_fits` is a thin wrapper over `_plan_spills(...).feasible`. It is strictly more permissive than the old depth-only check because spilling a long-lived multi-user value to a register removes it from the abstract stack.
- `GraphPartitioner._required_globals = ext_io ∪ overflow` — multi-user intermediates that fit in registers don't get a tensor slot, freeing both `max_tensors` budget and global memory traffic.
- `StackCodegen` accepts the spill plan via `node_to_reg`. `emit_op` emits `FStoreReg<N>` after computing a spilled node (and pops the value from the virtual stack); `handle_operand` emits `FLoadReg<N>` whenever an arg is needed and isn't on the virtual stack at depth 0/1. The generic-arg, depth-1, and depth-≥2 branches all prefer a register reload over a global reload.
- The host's `select_variant` automatically picks `l12` when the bytecode references registers (`max_reg_idx ≥ 4`); subgraphs with no spills stay on `s7`.
- Geometry_smith post-reduction tail: 14-node pointwise chain (two `schlick_ggx` calls + final multiply) used to split into a 10-node + 4-node pair (stack overflow at op 11). With register spills, all 14 ops fuse into a single 21-instruction kernel using regs 0 and 1 for `relu_1` and `k₁`. Wall time at M=1M: 115 µs → 77 µs (eager 102 µs).

#### Metadata Ops Support
- The conductor supports shape/stride-only ops (`view`, `unsqueeze`, `squeeze`, `expand`, `permute`, `transpose`, `t`, `slice`) as **identity on the stack** — no bytecode emitted, value passes through unchanged
- Registered as `OpKind.METADATA` in `op_registry.py` with `arity=1`, `arg_order=[0]` (only the tensor arg is pushed; shape/dim args are read from the FX node, not the stack)
- Enables fusion through metadata ops: e.g., `relu(x).unsqueeze(0) + 1.0` compiles to a single kernel instead of relu → eager unsqueeze → add (two kernels)
- Shape changes are handled by the broadcast plan's per-tensor `ProgramTensorInfo` strides — unsqueeze adds a size-1 dim that broadcasts naturally, expand sets stride=0 for expanded dims
- **Slice support** (`aten.slice.Tensor`): slice + pointwise fuses into a single kernel (e.g., `x[:, :64].relu()`). The slice input global uses effective shapes (sliced size). Non-zero start offsets are handled via `narrow()` at runtime (`input_adjustments` in `GintCompiledSubgraph`). Stepped slices (step != 1) are not fused.
- **`_get_strides`**: Reads actual tensor strides from FX metadata (`node.meta['val'].stride()`). Falls back to C-contiguous computation if unavailable. Strides are threaded through `_compute_broadcast_plan` for correct `block_stride` and `batch_strides` on non-contiguous tensors.
- **Partitioner safety**: When a metadata op's global input shape is not broadcast-compatible with the output shape (e.g., `view(1024) → (32, 32)`), the node is skipped and falls back to eager execution. The `_compute_broadcast_plan` validates that each tensor dim is either 1 or equal to the output dim. Slice ops are forced to start a new subgraph (they shrink a dimension which isn't broadcast-compatible).
- **Current limitation**: Non-contiguous input strides (e.g. transpose) are correctly read and used for the input tensors, but the output tensor is still created as C-contiguous. The output stride doesn't match the input stride for transposed views, so `x.t().relu()` still falls back to eager.

#### Reduction Support (sum, mean, prod, amax, amin)
- Registered as `OpKind.REDUCTION` in `op_registry.py` for `aten.sum.dim_IntList`, `aten.mean.dim`, `aten.prod.dim_int`, `aten.amax.default`, `aten.amin.default`
- **Constraint**: Innermost-dim only, single reduction dim. Non-innermost reductions fall back to eager
- **Tile dispatch (`_select_reduction_tiling`)**: Each warp packs `B * R == 128` lanes covering `B` batches × `R` reduction elements. Choice driven by innermost dim N:
  - `N >= 128` → (B=1, R=128, mode `1d`); threads+width flatten over reduction; final = warp_allreduce + width-combine.
  - `16 <= N < 128` → (B=4, R=32, mode `2dt`); threads scan reduction, width-lanes carry 4 batches; final = warp_allreduce only.
  - `N < 16` → (B=32, R=4, mode `2dw`); threads carry 32 batches, width-lanes scan reduction; final = width-combine only.
  - Both `_compile_reduction_subgraph` and `_compile_fused_reduction_subgraph` use this dispatch. The small-N (mode != '1d') fused path supports all three phase combinations — pre_prefix, scalar_prefix, and broadcast_suffix. Mode '1d' (large N) stays on the legacy fused codegen because its layout (batches in `batch_shape`, not `block_grid`) is structurally different and well-tested by RMSNorm
- **`combine_fn` / `warp_reduce_fn` / `post_reduce_fn`**: descriptor splits the reduction's emit into pieces so the compiler can call only the relevant ones per tile. `combine_fn` is the pairwise multi-chunk accumulator (fadd for sum/mean, fmul for prod, `dup2; fgt/flt; fselect` pair for amax/amin). `warp_reduce_fn` is the 1-instr `warp_allreduce_*`. `post_reduce_fn(node)` is the optional final fix-up (mean's `* 1/N`)
- **OOB padding caveat**: kernel pads OOB lanes with `0.0`. Sum/mean tolerate this; prod (× 0), amax (max-with-0 clamps negatives), amin (min-with-0 clamps positives) do NOT. Those three are gated by `_check_reduction_feasible_clean_chunks`, which now requires `N % R == 0` for the chosen tile (was `N % 128 == 0`). E.g. N=64 with tile R=32 is now accepted; previously rejected
- **No kernel changes required** — width-lane combining is composed from existing instructions:
  ```
  warp_allreduce_fsum     ; [p0, p1, p2, p3] per thread
  dup; fperm_w(2,3,0,1); fadd   ; [p0+p2, p1+p3, ...]
  dup; fperm_w(1,0,3,2); fadd   ; [total, total, total, total]
  ```
  Factored as `_emit_width_combine(combine_fn)` and reused per-tile by `_compile_reduction_subgraph`. Mean appends `fmulimm(1/N)` via `post_reduce_fn`
- **Batch-dim merge (`_merge_dims`)**: greedy incremental merge of consecutive batch dims that are C-contiguous in *every* tensor of the subgraph. Unlike `_compute_broadcast_plan`'s all-or-nothing merge, this stops at non-contiguous boundaries without collapsing back to length-1, so partial merges still help
- **Per-warp tile clamping**: the innermost merged batch dim is placed in `block_grid_dim_2` (modes 1d/2dt) or `block_grid_dim_1` (mode 2dw) with step=B, so the kernel's `b_shape - b_idx*step` clamping handles partial last tiles when `M % B != 0`. Outer batch dims stay in `batch_shape`
- **Multi-chunk loads**: For reduction dim > R, loads are unrolled at Python codegen time: `fldg_1d/2dt/2dw(0, slot)`, `(R, slot)`, `(2R, slot)`, … Each followed by `combine_fn`. Practical for N up to ~16K
- **Partitioner**: Reduction nodes are isolated as single-node subgraphs (flush any current pointwise subgraph)

#### Fused Reduction+Broadcast+Pointwise
- When a `keepdim=True` reduction is followed by a pointwise subgraph, the conductor fuses them post-partitioning into a single subgraph
- **Generalized multi-op support**: The post-reduction pointwise chain is split into a **scalar prefix** (ops consuming only the reduction scalar, e.g. `+eps`, `rsqrt`) and a **broadcast suffix** (ops consuming external globals, e.g. `*x`, `*w`). The scalar prefix executes once; the broadcast suffix runs per-chunk
- **Pre-reduction fusion**: A pointwise subgraph immediately preceding a reduction (e.g. `x*x` before `mean`) is absorbed into the Phase 1 accumulation loop — no intermediate global memory round-trip. `_split_pre_prefix(pre_schedule, reduction_input)` further splits pre_prefix into:
  - **per_chunk**: ancestors of the reduction input (e.g. `mul_1 = n*L` feeding `sum`) — emitted inside Phase 1's chunk loop, external inputs must be reduction-dim-shaped
  - **per_batch**: ops independent of the reduction chain that operate on `[..., 1]`-shaped per-batch values (e.g. `relu(sum_1)` running alongside an unrelated `n*L → sum_2` reduction) — emitted ONCE in Phase 0 before the chunk loop, external inputs broadcast across chunk lanes (stride 0 in inner axis), external outputs stored once via `fstg_2dt` writing only the size-1 axis. Per-batch external inputs and outputs are added to `scalar_globals` so their `ProgramTensorInfo` gets the size-1 inner stride
- **Four-phase bytecode** (mirrors the handwritten rmsnorm kernel in `tests/test_rmsnorm.py`):
  - Phase 0: per_batch_pre — load per-batch globals (`block_load_fn(0, slot)` with stride-0 inner), apply ops, store outputs via `scalar_store_fn`
  - Phase 1: For each chunk, load + apply per_chunk_pre → accumulate, then warp_allreduce + width-lane combine → scalar on stack
  - Phase 1b: Emit scalar prefix ops on the scalar
  - Phase 2: For each chunk, `dup` scalar, load chunked globals, emit broadcast suffix ops, store output
- **Two codegen paths**:
  - `_compile_fused_reduction_new_tile` for small N (modes 2dt and 2dw): tile-aware Phase 1 (chunked `fldg_2dt/2dw` + pre_prefix ops + accumulate), Phase 1b (scalar prefix on per-thread/per-width-lane scalar), and Phase 2 (per-chunk broadcast suffix — `dup` scalar, load external block-shaped globals via `block_load_fn(k*R, slot)`, run pointwise ops, store via `block_store_fn(k*R, slot)`). Per-tensor TensorInfo distinguishes "scalar" globals (single-element-per-batch — only the reduction-only output when broadcast_suffix is empty) from "block" globals (full reduction-dim tile, shape ending in N — every other global). Per-tensor batch strides pad with 0s on the left for tensors with fewer batch dims, and use 0 within-shape for size-1 broadcast dims; `_merge_dims` runs over all tensors jointly so partial merges are honored
  - `_compile_fused_reduction_subgraph` (legacy 1d) for mode '1d' only (N ≥ 128). Structurally distinct (batches in `batch_shape` vs `block_grid_dim_2`) and production-tested by RMSNorm
- **Geometry_smith fused-reduction** at N=3 (mode 2dw, pre_prefix split into per_batch=[relu]+per_chunk=[mul_1], no suffix): collapses the `relu(sum_1) + (n*L).sum(-1)` middle pair into one kernel, dropping the outer per-iter from 4 kernels (75 µs) to 3 kernels (63 µs) at M=1M. **Small-N RMSNorm-shape** at N=8 or N=64 (mode 2dw / 2dt, pre_prefix=[mul] + scalar_prefix=[+eps, rsqrt] + broadcast_suffix=[*x, *w]): now produces a single fused kernel covered by `tests/test_conductor_real_workloads.py::test_rms_norm_small_N_2dw / _2dt`
- **Fusion detection**: `_can_fuse_with_reduction` (post-reduction) and `_can_fuse_pre_reduction` (pre-reduction) validate that all involved ops are ELEMENTWISE, that the reduction result has no external users, and that all external globals share the reduction dimension
- **Deferred compilation**: Pointwise subgraphs preceding reductions are buffered rather than compiled immediately, avoiding wasted work when they are absorbed by pre-reduction fusion
- **Stack depth**: Phase 1 peaks at ~2 extra slots (width-lane combine). Phase 2 needs scalar + 1 for input + op overhead. Total ~4-5, well within MAX_STACK=8
- Examples: `x - mean(x)` (single binary op, existing pattern). `x * rsqrt(mean(x*x) + eps) * w` (full RMSNorm, 6 nodes fused into 1 kernel)

#### Reduction Future Work
- **Unify mode '1d' onto the new tile dispatch** — currently the legacy `_compile_fused_reduction_subgraph` handles mode '1d' (N ≥ 128) with batches in `batch_shape` and per-tensor block_size, while the new path uses `block_grid` and joint-merged batches. Migrating mode '1d' to the new path would remove the legacy codegen and unify the three modes; needs careful testing against RMSNorm
- **Dedicated `WarpFullReduceSum` instruction** — single opcode replacing the 7-instruction width-lane combine sequence
- **Kernel loop instruction** — for reduction dims > ~16K without bytecode bloat
- **Non-innermost reductions** — would require transposing the tensor or a different grid decomposition
- **Multi-reduction patterns** — e.g., layer norm (mean + variance in one pass)

## Dependencies

Runtime dependencies (declared in `pyproject.toml`):
- `numpy`, `llvmlite>=0.43`, `cuda-bindings>=12.6`, `rich`
- Optional: `torch>=2.0` (for conductor / `torch.compile` backend)
- Optional: `hip-python>=6.0` (for AMD HIP backend, see installation section below)

### cuda-python / cuda-bindings Version Compatibility

The `cuda-python` package was restructured into a metapackage starting at v12.6. The actual bindings live in `cuda-bindings`. Both 12.x (up to 12.9.x) and 13.x lines are actively maintained in parallel.

| Version line | Import style | Status |
|---|---|---|
| cuda-python <=12.5 | `from cuda import cuda` (old trampoline) | EOL |
| cuda-bindings 12.6–12.9.x | `import cuda.bindings.driver as cuda` | Maintained |
| cuda-bindings 13.0+ | `import cuda.bindings.driver as cuda` | Current |

**This project uses `import cuda.bindings.driver as cuda`**, which works on both 12.6+ and 13.x. The dependency spec `cuda-bindings>=12.6` accepts either line.

Key 13.0 changes: old trampoline modules (`cuda.cuda`, `cuda.cudart`) removed; GIL released for all C API calls; `int(cuda_obj)` deprecated in favor of `cuda.bindings.utils.get_cuda_native_handle()` (13.0+ only, `int()` still works). We use `int()` in `driver.py` and `executor_impl.py` — no change needed until `int()` is actually removed.

### GPU Architecture Support (Fatbin)

The fatbin (`gint/host/cuda/gint.fatbin.xz`) includes native SASS for:

| Architecture | Compute Capability | GPUs |
|---|---|---|
| Turing | sm_75 | RTX 2080, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090, L40 |
| Hopper | sm_90 | H100, H200 |
| Blackwell (datacenter) | sm_100 | B100, B200, GB200 |
| Blackwell (consumer) | sm_120 | RTX 5090, RTX 5080 |

Plus embedded `compute_120` PTX for forward compatibility with future architectures.

**Important**: sm_100 and sm_120 are **sibling** architectures, not parent/child. sm_120 is NOT a superset of sm_100 — they have different tensor core models. Both need explicit `-gencode` entries. PTX forward compatibility from `compute_90` is unreliable on Blackwell (the driver JIT can reject older PTX despite it being theoretically supported). Native SASS via CUDA 12.8+ is required.

**Build requirement**: `generate.sh` requires **CUDA Toolkit 12.8+** (for sm_100/sm_120 support). Earlier nvcc versions will fail on the Blackwell gencode flags.

### AMD GPU Architecture Support (HIP)

The AMDGCN fat binary (`gint/host/hip/gint.fatbin.xz`) bundles per-target HSACO code objects for:

| Architecture | GFX Target | GPUs |
|---|---|---|
| RDNA3 (discrete) | gfx1100, gfx1101 | RX 7900 XTX, RX 7800 XT, RX 7600 |
| RDNA4 | gfx1200, gfx1201 | RX 9070 XT, RX 9070 |

The fat binary uses the `__CLANG_OFFLOAD_BUNDLE__` format produced by `clang-offload-bundler`. `hipModuleLoadData` auto-selects the correct code object for the current GPU at runtime — no manual arch lookup needed.

**Build requirement**: `generate_amdgcn.sh` requires ROCm 7.0+ with OCML/OCKL bitcode libraries.

**CDNA not supported**: MI250/MI300 use wave64 (64-thread wavefronts), which would require 6-round reductions instead of 5, different shuffle masks, and changes to `lane_id` bounds. Only RDNA3+ (wave32) is supported.

### Cross-Platform Porting Notes

The architecture cleanly separates platform-specific code into two layers: `PlatformIRBuilder` (device-side, in `gint/kernel/platforms/`) and `BaseExecutor` (host-side, in `gint/host/`). All 106 opcodes and the entire interpreter dispatch loop are platform-agnostic LLVM IR. A new platform needs:

1. **`PlatformIRBuilder` subclass** — implement ~17 abstract methods: thread/warp indexing, warp shuffle/broadcast/reduce, shared memory address space, transcendental math library calls, kernel module creation (triple, data layout, calling convention)
2. **`BaseExecutor` subclass** — device memory management, kernel loading, launch API
3. **Build pipeline** — LLVM IR → platform binary (fatbin, HSACO, SPIR-V, etc.)

| Target | Difficulty | Notes |
|---|---|---|
| **AMD ROCm (HIP)** | **Implemented** | RDNA3+ (wave32) fully supported. `AMDGCNIRBuilder` in `gint/kernel/platforms/amdgcn.py`, `HipExecutor` in `gint/host/hip/`. Uses `llvm.amdgcn.*` intrinsics, `ocml` for math, `hip-python` for host API, `ds.bpermute` for warp reductions, `readlane` for broadcasts. CDNA (MI250/MI300, wave64) is NOT supported — would need 6-round reductions and different shuffle masks. |
| **OpenCL SPIR-V** | Moderate-Hard | Best cross-vendor path. `llvm-spirv` translates LLVM IR → OpenCL SPIR-V, and the OpenCL execution model (kernel functions with pointer args, flat address space, raw pointer arithmetic) matches our design closely. Subgroup ops (`cl_khr_subgroups`) map to warp primitives. Runs on Intel, AMD, NVIDIA via their OpenCL drivers. `pyopencl` for host API. Main issues: subgroup size varies by vendor (32/64/8-32), driver optimization quality lags CUDA. |
| **Vulkan SPIR-V** | Hard | Vulkan uses a fundamentally different execution model: no raw pointers (descriptor sets + buffer offsets), `GLCompute` execution model vs `Kernel`. Our pointer-table indirect dispatch doesn't translate directly. Would likely need a parallel SPIR-V codegen path rather than reusing llvmlite → `llvm-spirv`. Subgroup ops available (Vulkan 1.1+) but size varies. Math via `GLSL.std.450` extended instructions. |
| **Apple Metal GPU** | Hard | No LLVM IR → Metal path. Would need to generate MSL source or Metal IR directly. Metal has `simdgroup` ops (size 32 on Apple Silicon) that map well to our warp primitives. `pyobjc` for host API. Apple's `metal` compiler is clang-based but Metal IR format is undocumented. |
| **Apple NPU (ANE)** | Impossible | Not a programmable compute device — executes compiled neural network graphs (Core ML), no user-accessible ISA, no thread model. |

**Key constraint across all targets**: the interpreter assumes `WARP_SIZE=32` (32 threads per subgroup). RDNA3 AMD GPUs support wave32 natively, but CDNA and some Intel GPUs use 64 or variable sizes. This affects `lane_id` bounds, shuffle masks, and the tree reduction in `warp_allreduce` (5 rounds for 32, 6 for 64).

## Common Commands

### Testing
```bash
# Run all tests with coverage
python run_tests.py

# Run specific test module
python run_tests.py tests.test_frontend

# Coverage is generated in htmlcov/
```

Tests use Python's `unittest` framework (not pytest).

### Code Generation
```bash
# NVIDIA: Generate LLVM IR and compile to PTX/fatbin for all compute capabilities
# Requires: CUDA Toolkit 12.8+, LLVM/clang 20, llvmlite, rich, numpy
./generate.sh

# Generates files:
# - artifact/gint.ptx
# - artifact/gint.fatbin
# - artifact/gint.fatbin.xz (compressed, used by runtime)
# - gint/host/cuda/gint.fatbin.xz (deployed version, NOT tracked in git)

# AMD: Generate LLVM IR and compile to HSACO for RDNA3+ targets
# Requires: ROCm 7.0+, OCML/OCKL bitcode
./generate_amdgcn.sh

# Generates files:
# - artifact/gint_gfx*.s (per-target GCN assembly text)
# - artifact/gint_gfx*.o (per-target code objects)
# - artifact/gint_amdgcn.fatbin (bundled fat binary, all targets)
# - gint/host/hip/gint.fatbin.xz (deployed version, NOT tracked in git)

# Direct LLVM IR generation (for debugging/inspection)
gint-gen-llir -t llir      # NVPTX LLVM IR (default)
gint-gen-llir -t amdgcn --gfx gfx1100  # AMDGCN assembly
```

### CI / Wheel Build

GitHub Actions workflow (`.github/workflows/build.yml`) — **manual dispatch only** (`workflow_dispatch`):
1. `generate-fatbin` job: runs in `nvidia/cuda:12.8.1-devel-ubuntu22.04` container, installs LLVM 18, generates the fatbin
2. `build-wheel` job: downloads fatbin artifact, builds a `py3-none-any` wheel with hatchling, verifies fatbin is included

The wheel is pure Python (no native extensions) so only one Python version is needed for the build. The fatbin is included via hatchling's `artifacts` config which overrides `.gitignore` exclusion.

## Key Implementation Details

### Data Model
- **`TensorInfo` (Device) / `HTensorInfo` (Host)**: Shared struct for tensor metadata (base pointers, strides, shapes)
- Used for multi-dimensional data access patterns on the device

### Bytecode Instruction Format
- Each instruction is a 2-word pair `[opcode: i32, operand: i32]` in the bytecode stream
- PC always advances by 2 after each instruction
- Instructions with no meaningful operand use `operand = 0`
- Opcodes defined in `INSNS` dict in `gint/kernel/interpreter/main.py`

### Kernel Signature
```
void geval(i32* code, TensorInfo* tinfo, i32 num_tensors, i32 grid_dim, i32 flag)
```
- `flag <= 0`: direct mode — `code` and `tinfo` are used as-is by all warps
- `flag > 0`: indirect mode — `code` and `tinfo` are cast to `i32**` / `TensorInfo**` and indexed per-warp

### Kernel Specialization
- Per-vendor implementations in `gint/kernel/platforms/`
  - `nvptx.py`: NVIDIA — `llvm.nvvm.*` intrinsics, `__nv_*` libdevice math, `nvptx64-nvidia-cuda` triple
  - `amdgcn.py`: AMD RDNA3+ — `llvm.amdgcn.*` intrinsics, `__ocml_*_f32` math, `amdgcn-amd-amdhsa` triple
- **llvmlite limitation**: `FunctionAttributes.add()` doesn't support key-value attributes (e.g., `"amdgpu-flat-work-group-size"="32,128"`). The AMDGCN builder works around this by overriding `emit()` to inject LLVM attribute groups via regex post-processing of the serialized IR text
- Per-OS optimizations may be needed in executor implementations

### Adding New Instructions
1. Implement the instruction class in `gint/kernel/interpreter/instructions/` (subclass `DefaultControlInstruction` or `DefaultControlOperandInstruction`)
2. Add to `INSNS` dict in `gint/kernel/interpreter/main.py` with the next available opcode
3. Add a frontend function in `gint/host/frontend.py` decorated with `@_bc`
4. Run `./generate.sh` (NVIDIA) and/or `./generate_amdgcn.sh` (AMD) to regenerate binaries

## Project Layout

```
gint/
  host/              # Host-side Python + CUDA/HIP bindings
    cuda/            # NVIDIA-specific (executor_impl, driver)
                     # gint.fatbin.xz lives here but is NOT in git
    hip/             # AMD-specific (executor_impl, driver)
                     # gint.fatbin.xz lives here but is NOT in git
    frontend.py      # @bytecode decorator & ProgramTensorInfo
    executor.py      # Base executor interface + backend auto-detection
    utils.py         # Shared utilities (fill_tensor_info, cdiv)
    sugar.py         # Convenience APIs
    analyzer.py      # Static stack/reg usage analyzer for recorded bytecode
  kernel/            # Device-side LLVM IR code
    interpreter/     # Stack-based VM implementation
      instructions/  # Instruction definitions (arith, control, load_store, reg, ...)
      state.py       # StackMachineState: unified pool + stack/reg properties
      main.py        # INSNS opcode table, build_main_loop, constants
    platforms/       # Platform-specific code (nvptx.py, amdgcn.py)
  conductor/         # torch.compile backend
  scripts/           # gen_llir.py, driver.py

examples/
  superopt/          # GPU-accelerated bytecode superoptimizer

tests/               # unittest modules
benchmark/           # Performance benchmarks
artifact/            # Build outputs (not in git)
```

### Superoptimizer
```bash
# Search for shorter equivalent bytecode sequences for a target
python -m examples.superopt relu

# Run on all targets
python -m examples.superopt --all

# List available targets
python -m examples.superopt --list
```

## Superoptimizer (`examples/superopt/`)

A GPU-accelerated superoptimizer that finds shorter equivalent bytecode sequences for gint instruction patterns. Uses `execute_indirect` to evaluate thousands of candidate programs per kernel launch — no kernel changes needed.

### Architecture
- **`opcodes.py`**: Search space definition — 33+ ops with stack effects (min_depth, net_effect). Expandable with transcendentals.
- **`candidates.py`**: DFS enumeration with depth-reachability pruning. For length 5 unary targets, prunes 33^5=40M down to ~1.5M valid sequences. Also has numpy-vectorized random batch generation for stochastic search.
- **`executor.py`**: `BatchRunner` — concatenates all candidate bytecodes into one device allocation, builds per-candidate tensor infos with only the output pointer patched, uses a single indirect-mode kernel launch. ~8 CUDA API calls regardless of batch size.
- **`search.py`**: Brute-force (exhaustive, length 1..N) and stochastic (random mutation hill climbing) modes. Candidates verified with 10 additional random test vector sets.
- **`targets.py`**: Reference sequences from `op_registry.py` to optimize.

### Key insight discovered
`fselect` treats `peek(0)` as a float condition (`>0` → true branch). This means explicit comparison-against-zero patterns (`fpush(0) → flt/fgt → fselect`) can often be eliminated:
- **relu**: `x * float(x > 0)` via `dup → fpush(0) → flt → fmul` (4 insns, was 5)
- **abs**: `select(-x > 0, -x, x)` via `dup → fneg → dup → fselect` (4 insns, was 7)
- **leaky_relu**: `dup → fmulimm(ns) → dupx1 → fselect` exploits that `ns*x > 0 ↔ x > 0` for positive slopes (4 insns, was 7)
- **gelu, silu**: Already optimal (6 and 5 insns respectively)

## Installation

Built with `hatchling`. Requires Python >=3.10.

```bash
pip install -e .          # core (needs cuda-bindings, numpy, llvmlite, rich)
pip install -e ".[torch]" # with torch.compile backend

# AMD HIP backend (hip-python is only on Test PyPI due to AMD's packaging policy)
python3 -m pip install -i https://test.pypi.org/simple "hip-python>=6.0.0"
```

> **Note**: `hip-python` is not declared as a pip dependency because AMD only publishes real packages to Test PyPI (the PyPI entry is a dummy). This needs to be documented for end users when gint is published.

Entry points:
- `gint-gen-llir`: LLVM IR generation script
- `gint-driver`: Driver utility
