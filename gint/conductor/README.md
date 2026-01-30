# Gint Conductor - Torch.Compile Backend

The `conductor` submodule provides integration between PyTorch's `torch.compile` infrastructure and the gint GPU interpreter.

## Overview

This module implements a custom backend for `torch.compile`, allowing PyTorch models to be compiled into gint bytecode and executed on the GPU using gint's device-side interpreter.

## Architecture

The conductor module consists of several components:

### 1. Backend (`backend.py`)
- **`gint_backend()`**: Main entry point that implements the torch.compile backend contract
- **`register_backend()`**: Utility to register gint with torch._dynamo

### 2. Compiler (`compiler.py`)
- **`GintCompiler`**: Converts FX graphs to gint bytecode
- **`GintCompiledSubgraph`**: Executable program wrapper for compiled bytecode
- Handles operation mapping and bytecode generation
- Manage execution of subgraphs and fallback to eager mode

### 3. Partitioner (`partitioner.py`)
- **`GraphPartitioner`**: Partitions FX graphs into subgraphs compatible with gint constraints
    - **Constraints**: Max 8 tensors per subgraph, uniform shape, supported operators only

## Usage

```python
import torch
from gint.conductor import register_backend

# Register the gint backend
register_backend()

# Use torch.compile with gint backend
@torch.compile(backend="gint")
def my_function(x, y):
    return x + y * 2

# Execute
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
result = my_function(x, y)
```

## Current Status

This is a proof-of-concept implementation with the following features:

### ✅ Implemented
- Basic backend registration
- Graph partitioning to respect device interpreter limits (max 8 tensors)
- Operation mapping for basic arithmetic (`add`, `sub`, `mul`, `div`)
- Stream synchronization with PyTorch
- Fallback to eager execution for unsupported operators or graph structures

### 🚧 TODO
- [ ] Expand operator support (unary math, comparisons, reductions)
- [ ] Support mixed shapes within subgraphs
- [ ] Optimize memory access patterns (coalescing)
- [ ] Dynamic shape support

## Integration with Gint

The conductor module builds on gint's existing infrastructure:

- **Frontend API** (`gint.host.frontend`): Provides bytecode instruction builders
- **Executor** (`gint.host.executor`): Handles program execution on GPU
- **Interpreter** (`gint.kernel.interpreter`): Device-side bytecode interpreter
