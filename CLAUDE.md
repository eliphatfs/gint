# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Gint** is an experimental project implementing a **completely GPU device-side interpreter architecture** for kernel programming with cross-OS and cross-device-vendor support. The project is written with Python + LLVM IR, enabling efficient bytecode execution directly on the GPU.

The codebase is split into:
- **Host side**: Python + CUDA C bindings for device memory management, kernel loading, and bytecode dispatch
- **Kernel side**: LLVM IR interpreter executing stack-based bytecode on GPU
- **Conductor**: PyTorch `torch.compile` backend integration

## Architecture

### Stack-Based Interpreter (Device-Side)
- Located in `gint/kernel/interpreter/` - written in LLVM IR via `llvmlite`
- **Micro-architecture**:
  - `REG_WIDTH = 4`: Processes 4 elements simultaneously (ILP - Instruction Level Parallelism)
  - `POOL_SIZE = 12`, `NUM_REGS = 8`, `MAX_STACK = 8`: Unified pool of 12 slots — stack grows upward from index 0, virtual registers occupy the top 8 slots counting downward (`reg n = pool[11-n]`). The upper 4 stack slots overlap with the lower 4 registers, so kernels must not simultaneously maximize both stack depth and register usage.
  - Dispatch: Large switch-case in LLVM IR with weights for optimization
  - Uses PHI nodes for state persistence across instruction dispatches (one dispatch block per stack depth 0..MAX_STACK)
- **Instruction set**: Defined in `gint/kernel/interpreter/instructions/`

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
- `gint/host/executor.py`: Core program execution interface
- `gint/host/cuda/executor_impl.py`: NVIDIA-specific implementation (CudaExecutor)
- Loads kernel from `gint/host/cuda/gint.fatbin.xz` (not tracked in git — must be generated locally via `./generate.sh`)
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

### Torch.Compile Integration (Conductor)
- `gint/conductor/backend.py`: Backend registration and entry point
- `gint/conductor/compiler.py`: FX graph → bytecode conversion, graph partitioning, and broadcasting
- Supports basic arithmetic ops (add, sub, mul, div), unary transcendentals, activations (relu, gelu, silu, leaky_relu), comparisons, `where`, and metadata ops (view, unsqueeze, squeeze, expand, permute, transpose, t); fallback to eager mode for unsupported ops
- Partitioner constraints per subgraph: max 8 global tensor slots, max stack depth 8, broadcast-compatible shapes

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

#### Metadata Ops Support
- The conductor supports shape/stride-only ops (`view`, `unsqueeze`, `squeeze`, `expand`, `permute`, `transpose`, `t`) as **identity on the stack** — no bytecode emitted, value passes through unchanged
- Registered as `OpKind.METADATA` in `op_registry.py` with `arity=1`, `arg_order=[0]` (only the tensor arg is pushed; shape/dim args are read from the FX node, not the stack)
- Enables fusion through metadata ops: e.g., `relu(x).unsqueeze(0) + 1.0` compiles to a single kernel instead of relu → eager unsqueeze → add (two kernels)
- Shape changes are handled by the broadcast plan's per-tensor `ProgramTensorInfo` strides — unsqueeze adds a size-1 dim that broadcasts naturally, expand sets stride=0 for expanded dims
- **Partitioner safety**: When a metadata op's global input shape is not broadcast-compatible with the output shape (e.g., `view(1024) → (32, 32)`), the node is skipped and falls back to eager execution. The `_compute_broadcast_plan` validates that each tensor dim is either 1 or equal to the output dim
- **Current limitation**: Transpose/permute are registered but typically cause subgraph breaks because the transposed shape isn't broadcast-compatible with the original. Full support would require the broadcast plan to use actual tensor strides from FX metadata instead of computed C-contiguous strides

#### Reduction Design Notes (planned, not yet implemented)
- **Existing kernel primitives**: `warp_allreduce_fsum/fmax/fmin/fprod` reduce across 32 threads, independently per width lane (4 partial results). Need width-lane reduce + loop for full reductions.
- **Missing kernel pieces**: (1) Width-lane reduction instruction to combine 4 partial sums into one scalar. (2) Loop instruction for reduction dims > 128 elements.
- **PyTorch's approach** for large vector reductions (`reduce_kernel`): Launches many blocks (all SMs active). Each block reduces its chunk, writes partial to a global buffer, then `atomicInc` a completion counter. The **last block** to finish detects this via the counter and does the final reduction of all partials. This packs a two-pass tree reduction into a single kernel launch. The atomic is on a **counter** (not the output value), avoiding FP non-determinism.
- **Gint approach options**:
  - **Two-program via `execute_indirect`**: Program 1 = each warp reduces a chunk and stores a partial. Program 2 = one warp reduces the partials. Already supported infrastructure, needs loop + width-lane reduce instructions only.
  - **Atomic single-pass**: Each warp reduces its chunk, `atomicAdd`s to output. Needs new `atomic_rmw` instruction (`LL.atomic_rmw('fadd', ptr, val, 'monotonic')` in llvmlite → `atom.global.add.f32` on sm_60+).
  - **Last-block trick** (like PyTorch): Needs `atomicInc` instruction + global partial buffer + conditional branch. Most efficient (single launch, full utilization, no FP atomics) but requires the most kernel additions.

## Dependencies

Runtime dependencies (declared in `pyproject.toml`):
- `numpy`, `llvmlite>=0.43`, `cuda-bindings>=12.6`, `rich`
- Optional: `torch>=2.0` (for conductor / `torch.compile` backend)

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

### Cross-Platform Porting Notes

The architecture cleanly separates platform-specific code into two layers: `PlatformIRBuilder` (device-side, in `gint/kernel/platforms/`) and `BaseExecutor` (host-side, in `gint/host/`). All 106 opcodes and the entire interpreter dispatch loop are platform-agnostic LLVM IR. A new platform needs:

1. **`PlatformIRBuilder` subclass** — implement ~17 abstract methods: thread/warp indexing, warp shuffle/broadcast/reduce, shared memory address space, transcendental math library calls, kernel module creation (triple, data layout, calling convention)
2. **`BaseExecutor` subclass** — device memory management, kernel loading, launch API
3. **Build pipeline** — LLVM IR → platform binary (fatbin, HSACO, SPIR-V, etc.)

| Target | Difficulty | Notes |
|---|---|---|
| **AMD ROCm (HIP)** | Moderate | Maps 1:1 conceptually. `llvm.amdgcn.*` intrinsics for shuffles/indexing, `ocml` for math (like libdevice). `hip-python` for host API. Main issue: wavefront size is 64 on CDNA (MI250/MI300) vs our assumed 32 — either force wave32 on RDNA3 or adapt shuffle masks and reduction rounds for wave64. Also unclear if llvmlite bundles the AMDGCN backend. |
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
# Generate LLVM IR and compile to PTX/fatbin for all compute capabilities
# Requires: CUDA Toolkit 12.8+, LLVM/clang 18, llvmlite, rich, numpy
./generate.sh

# Generates files:
# - artifact/gint.ptx
# - artifact/gint.fatbin
# - artifact/gint.fatbin.xz (compressed, used by runtime)
# - gint/host/cuda/gint.fatbin.xz (deployed version, NOT tracked in git)
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
- Per-OS optimizations may be needed in executor implementations

### Adding New Instructions
1. Implement the instruction class in `gint/kernel/interpreter/instructions/` (subclass `DefaultControlInstruction` or `DefaultControlOperandInstruction`)
2. Add to `INSNS` dict in `gint/kernel/interpreter/main.py` with the next available opcode
3. Add a frontend function in `gint/host/frontend.py` decorated with `@_bc`
4. Run `./generate.sh` to regenerate the fatbin

## Project Layout

```
gint/
  host/              # Host-side Python + CUDA bindings
    cuda/            # NVIDIA-specific (executor_impl, driver)
                     # gint.fatbin.xz lives here but is NOT in git
    frontend.py      # @bytecode decorator & ProgramTensorInfo
    executor.py      # Base executor interface
    sugar.py         # Convenience APIs
  kernel/            # Device-side LLVM IR code
    interpreter/     # Stack-based VM implementation
      instructions/  # Instruction definitions (arith, control, load_store, reg, ...)
      state.py       # StackMachineState: unified pool + stack/reg properties
      main.py        # INSNS opcode table, build_main_loop, constants
    platforms/       # Platform-specific code
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
```

Entry points:
- `gint-gen-llir`: LLVM IR generation script
- `gint-driver`: Driver utility
