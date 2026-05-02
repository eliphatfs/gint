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

## Documentation Index

For deep details, consult:
- `docs/kernel.md` — interpreter dispatch, kernel variants (`s7` / `l12`), virtual register file, width-lane shuffles, indirect dispatch, bytecode format, adding new instructions
- `docs/conductor.md` — torch.compile backend internals: backend registration, CUDA graphs, broadcasting, pointwise tile dispatch, register-spill codegen, CSE, commutative-op elision, metadata ops, reductions, fused reduction+broadcast+pointwise
- `docs/platforms.md` — host executor, NVIDIA / AMD GPU support tables, cuda-bindings compatibility, cross-platform porting notes
- `docs/superopt.md` — GPU-accelerated bytecode superoptimizer in `examples/superopt/`
- `docs/install.md` — install, dependencies, code generation, CI / wheel build

## Architecture (high level)

### Stack-Based Interpreter
- Located in `gint/kernel/interpreter/` — written in LLVM IR via `llvmlite`.
- `REG_WIDTH = 4`: each lane processes 4 elements simultaneously (ILP).
- Pool/stack/register sizes are per-variant. Stack grows upward from index 0; virtual registers occupy the top `NUM_REGS` slots counting downward (`reg n = pool[POOL_SIZE-1-n]`).
- Dispatch: large switch-case in LLVM IR with PHI nodes for state persistence (one dispatch block per stack depth `0..MAX_STACK`).
- See `docs/kernel.md` for the full instruction set, indirect dispatch, and virtual register file.

### Kernel Variants (most important invariant)
The fatbin/HSACO ships **two kernel variants** under symbols `geval_s7` and `geval_l12`:

| Variant | POOL_SIZE | NUM_REGS | MAX_STACK | Use |
|---|---|---|---|---|
| `s7` | 7 | 4 | 7 | All pointwise/streaming workloads. Better occupancy. |
| `l12` | 12 | 8 | 8 | Register-heavy kernels (e.g. `inv4x4`). |

- **Variant table**: `gint.kernel.interpreter.main.VARIANTS` maps name → `(pool_size, num_regs, max_stack)`.
- **Host-side selection** (`gint/host/executor.py:select_variant`): runs `analyze_bytecode`, picks the smallest variant that fits. Cached on `program._cu` / `program._hip`.
- **`execute_indirect`**: picks `max(variant)` over the batch (one launch).
- **Build/deploy**: `./generate.sh` and `./generate_amdgcn.sh` bundle BOTH variants. See `docs/kernel.md` for opcode-filtering details.

### Frontend API
- `gint/host/frontend.py`: Pythonic bytecode generation
  - `@bytecode` decorator: intercepts Python calls to record instruction sequences.
  - `ProgramTensorInfo`: encapsulates tensor metadata (strides, shapes, element sizes).
  - Each instruction emits a 2-word pair `[opcode, operand]`; operand is 0 for instructions that don't use it.
- `gint/host/sugar.py`: higher-level convenience functions.
- `gint/host/analyzer.py`: static analyzer over recorded bytecode — `analyze_bytecode(bc)` returns `BytecodeStats` (peak stack depth, registers used, min pool size). Torch-free; useful for variant selection. Re-exported as `gint.analyze_bytecode`.

### Torch.Compile Integration (Conductor)
- `gint/conductor/backend.py`: backend registration (`"gint"` default with CUDA graphs, `"gint-no-cuda-graph"` legacy alias). Also exposes `gint.conductor.compile` — a `torch.compile` drop-in that scopes `automatic_dynamic_shapes=False` to the wrapped call (no global config flip; required because gint compiles per-shape and can't accept SymInt FakeTensors).
- `gint/conductor/compiler.py`: FX graph → bytecode conversion, graph partitioning, broadcasting.
- `gint/conductor/debug.py`: `inspect_subgraphs(fn, *args)` + `print_subgraphs` for inspecting compiled subgraphs.
- Op surface (full list in `op_registry.py`): arithmetic, comparisons, transcendentals, activations, clamp, composite, metadata (view/unsqueeze/squeeze/expand/permute/transpose/t/slice), innermost-dim reductions (sum/mean/prod/amax/amin). Unsupported ops fall back to eager.
- Special-op rewrite (`gint/conductor/special_ops.py`): pre-partition pass replaces `aten.bmm.default` and `getitem(linalg_inv_ex(a), 0)` with calls to `gint.host.matrix.gint_bmm` / `gint_inv` for square matrices with N ≤ 4. The rewritten nodes run via the eager-fallback path (which dispatches to a `SugarProgram`); surrounding pointwise ops still go through gint codegen.
- Partitioner constraints per subgraph: max 8 global tensor slots, max stack depth 8, broadcast-compatible shapes.
- See `docs/conductor.md` for codegen details (broadcasting, tile dispatch, register spills, CSE, fused reductions, etc.).

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
./generate.sh           # NVIDIA fatbin (requires CUDA 12.8+)
./generate_amdgcn.sh    # AMD HSACO (requires ROCm 7.0+)
```
See `docs/install.md` for build outputs and direct LLVM IR generation.

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
    matrix.py        # Small batched (N<=4) bmm / inverse SugarPrograms + Python wrappers
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
  superopt/          # GPU-accelerated bytecode superoptimizer (see docs/superopt.md)

tests/               # unittest modules
benchmark/           # Performance benchmarks
artifact/            # Build outputs (not in git)
docs/                # Deep-dive references (kernel, conductor, platforms, superopt, install)
```

## Data Model

- **`TensorInfo` (Device) / `HTensorInfo` (Host)**: shared struct for tensor metadata (base pointers, strides, shapes). Used for multi-dimensional data access patterns on the device.
- **Bytecode**: each instruction is a 2-word pair `[opcode: i32, operand: i32]` in the bytecode stream. PC advances by 2. Opcodes defined in `INSNS` dict in `gint/kernel/interpreter/main.py`.
- **Kernel signature**: `void geval(i32* code, TensorInfo* tinfo, i32 num_tensors, i32 grid_dim, i32 flag)`. `flag <= 0` is direct mode; `flag > 0` is indirect (per-warp pointer tables). See `docs/kernel.md`.
