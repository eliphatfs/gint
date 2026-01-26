Project Overview for Gemini

This is an experimental project to explore completely GPU device-side interpreter architecture for kernel programming.
It aims for cross-OS, cross-device-vendor support.

The project is written with python LLVM IR emitter, and the generation script is in `scripts/gen_llir.py` and can be executed as `gint-gen-llir`.

The project is split into two parts: host and kernel. The kernel part is mainly used only by the generation script, and does not execute in user runtime except for several hyperparameters.

Special functions need to be implemented by kernel/platforms per-vendor. The host side is a bit trickier and requires an executor implementation for each vendor and possibly need specializations for each OS.

## Project Notes

### Architecture Overview
The project implements a **completely GPU device-side interpreter** for executing kernels. It follows a stack-based execution model where bytecode is interpreted directly on the device.
- **Host Side**: Responsible for managing device memory, loading kernels, and dispatching bytecode. The `CudaExecutor` (in `gint/host/cuda`) handles NVIDIA-specific interactions.
- **Kernel Side**: A stack-heavy interpreter (in `gint/kernel/interpreter`) written in LLVM IR (via `llvmlite`). It uses PHI nodes for state persistence across instruction dispatches to maximize performance.

### Frontend API
The frontend (in `gint/host/frontend.py`) provides a Pythonic way to define GPU kernels:
- **`@bytecode` decorator**: Intercepts Python calls to record a sequence of instructions, which are then compiled into gint-compatible bytecode.
- **`ProgramTensorInfo`**: Encapsulates tensor layout metadata (strides, shapes, element sizes) for the device-side interpreter.

### Micro-Architecture
- **ILP (Instruction Level Parallelism)**: The interpreter operates on blocks of `REG_WIDTH = 4` elements simultaneously.
- **Stack Machine**: Uses a stack (`MAX_STACK = 8`) for intermediate values. Each stack entry represents a vector of `REG_WIDTH` elements.
- **Dispatch**: Implementation uses a large switch-case in LLVM IR to dispatch opcodes, with weights used to optimize for likely/unlikely instructions.

### Data Model
- **`TensorInfo` (Device)** / **`HTensorInfo` (Host)**: Shared structure for passing tensor metadata, including base pointers and complex stride/shape information for multi-dimensional data access.
