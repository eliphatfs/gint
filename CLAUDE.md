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
  - `MAX_STACK = 8`: Stack-based execution model with 8 stack slots, each holding a vector
  - Dispatch: Large switch-case in LLVM IR with weights for optimization
  - Uses PHI nodes for state persistence across instruction dispatches
- **Instruction set**: Basic arithmetic operations defined in `gint/kernel/interpreter/instructions/`

### Host-Side Executor
- `gint/host/executor.py`: Core program execution interface
- `gint/host/cuda/executor_impl.py`: NVIDIA-specific implementation (CudaExecutor)
- Manages kernel PTX loading from `gint/host/cuda/gint.fatbin.xz`

### Frontend API
- `gint/host/frontend.py`: Pythonic bytecode generation
  - `@bytecode` decorator: Intercepts Python calls to record instruction sequences
  - `ProgramTensorInfo`: Encapsulates tensor metadata (strides, shapes, element sizes)
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
# - gint/host/cuda/gint.fatbin.xz (deployed version)
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

### Kernel Specialization
- Per-vendor implementations in `gint/kernel/platforms/`
- Per-OS optimizations may be needed in executor implementations

### Bytecode Instruction Format
- Instructions compiled from Python-side definitions
- Mapped through `gint/conductor/op_registry.py` for torch.compile operations
- Stack machine semantics with vector operations

## Project Layout

```
gint/
  host/              # Host-side Python + CUDA bindings
    cuda/            # NVIDIA-specific (executor_impl, driver, fatbin)
    frontend.py      # @bytecode decorator & ProgramTensorInfo
    executor.py      # Base executor interface
    sugar.py         # Convenience APIs
  kernel/            # Device-side LLVM IR code
    interpreter/     # Stack-based VM implementation
      instructions/  # Instruction definitions
    platforms/       # Platform-specific code
  conductor/         # torch.compile backend
  scripts/           # gen_llir.py, driver.py

tests/               # unittest modules
benchmark/           # Performance benchmarks
```

## Installation

Built with `hatchling`. The project exposes:
- `gint-gen-llir`: LLVM IR generation script
- `gint-driver`: Driver utility
