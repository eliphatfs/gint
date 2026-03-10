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

### Host-Side Executor
- `gint/host/executor.py`: Core program execution interface
- `gint/host/cuda/executor_impl.py`: NVIDIA-specific implementation (CudaExecutor)
- Loads kernel from `gint/host/cuda/gint.fatbin.xz` (not tracked in git — must be generated locally via `./generate.sh`)

### Frontend API
- `gint/host/frontend.py`: Pythonic bytecode generation
  - `@bytecode` decorator: Intercepts Python calls to record instruction sequences
  - `ProgramTensorInfo`: Encapsulates tensor metadata (strides, shapes, element sizes)
  - Each instruction emits a 2-word pair `[opcode, operand]`; operand is 0 for instructions that don't use it
- `gint/host/sugar.py`: Higher-level convenience functions

### Torch.Compile Integration (Conductor)
- `gint/conductor/backend.py`: Backend registration and entry point
- `gint/conductor/compiler.py`: FX graph → bytecode conversion
- `gint/conductor/partitioner.py`: Graph partitioning with constraints (max 8 tensors, uniform shapes)
- Supports basic arithmetic ops (add, sub, mul, div); fallback to eager mode for unsupported ops

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

tests/               # unittest modules
benchmark/           # Performance benchmarks
artifact/            # Build outputs (not in git)
```

## Installation

Built with `hatchling`. The project exposes:
- `gint-gen-llir`: LLVM IR generation script
- `gint-driver`: Driver utility
