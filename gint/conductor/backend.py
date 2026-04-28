"""
Core torch.compile backend implementation for gint.

This module implements the backend contract required by torch.compile,
converting FX graphs into gint bytecode programs.
"""

import torch
from typing import List, Callable
from torch.fx import GraphModule

from ..host.executor import TensorInterface, get_executor
from .compiler import GintCompiler


def _make_gint_backend(cuda_graphs: bool) -> Callable:
    """Return a torch.compile backend callable with the cuda_graphs flag baked in."""
    def gint_backend(gm: GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
        from torch._functorch.aot_autograd import aot_module

        def compiler_fn(gm: GraphModule, example_inputs: List[torch.Tensor]):
            compiler = GintCompiler(gm, example_inputs)
            return compiler.compile()

        compiled = aot_module(gm, fw_compiler=compiler_fn)

        if not cuda_graphs:
            return compiled

        if not (torch.cuda.is_available() and all(
            isinstance(x, torch.Tensor) and x.is_cuda for x in example_inputs
        )):
            return compiled

        try:
            return torch.cuda.make_graphed_callables(compiled, tuple(example_inputs))
        except Exception as e:
            print(f"[gint] CUDA graph capture failed ({e!r}); falling back to non-graphed path")
            return compiled

    return gint_backend


def register_backend(name: str, cuda_graphs: bool):
    """
    Register a gint torch.compile backend under ``name``.

    On ``import gint.conductor`` the package auto-registers two backends:

    - ``"gint"`` — cuda_graphs=True (default).
    - ``"gint-no-cuda-graph"`` — cuda_graphs=False.

    Use this function only if you need a custom name or to re-register
    after a torch dynamo reset. Re-registering an existing name is a
    no-op (logged, not raised).

    Args:
        name: The name to register the backend under.
        cuda_graphs: If True, wrap the compiled forward in
            ``torch.cuda.make_graphed_callables`` so subsequent calls
            replay a CUDA graph.
    """
    backend_fn = _make_gint_backend(cuda_graphs=cuda_graphs)
    try:
        torch._dynamo.register_backend(name=name, compiler_fn=backend_fn)
    except AttributeError:
        raise RuntimeError(
            "torch._dynamo not available. "
            "Please ensure you have PyTorch 2.0+ installed with dynamo support."
        )
    except Exception as e:
        # Most commonly: name already registered (e.g. module re-import). Don't crash imports.
        print(f"[gint] register_backend({name!r}) skipped: {e!r}")
