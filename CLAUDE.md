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
./generate.sh

# Generates files:
# - artifact/gint.ptx
# - artifact/gint.fatbin
# - artifact/gint.fatbin.xz (compressed, used by runtime)
# - gint/host/cuda/gint.fatbin.xz (deployed version, NOT tracked in git)
```

**Manual generation steps** (if needed):
```bash
# Step 1: Generate LLVM IR and compile to PTX
gint-gen-llir -t ptx --cc 70 -o artifact/gint.ptx

# Step 2: Compile PTX to fatbin (multi-arch)
nvcc -fatbin --ptxas-options=-v \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  artifact/gint.ptx -o artifact/gint.fatbin

# Step 3: Compress and deploy
xz -efk artifact/gint.fatbin
cp artifact/gint.fatbin.xz gint/host/cuda/gint.fatbin.xz
```

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

Built with `hatchling`. The project exposes:
- `gint-gen-llir`: LLVM IR generation script
- `gint-driver`: Driver utility
