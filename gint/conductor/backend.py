"""
Core torch.compile backend implementation for gint.

This module implements the backend contract required by torch.compile,
converting FX graphs into gint bytecode programs.
"""

import os
import torch
from typing import List, Callable, Optional
from torch.fx import GraphModule

from ..host.executor import TensorInterface, get_executor
from .compiler import GintCompiler


def _resolve_cuda_graphs(flag: Optional[bool]) -> bool:
    if flag is not None:
        return bool(flag)
    return os.environ.get('GINT_CUDA_GRAPHS', '').lower() in ('1', 'true', 'yes', 'on')


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


# Public default backend (no graphs); preserved for users importing it directly.
gint_backend = _make_gint_backend(cuda_graphs=False)


def register_backend(name: str = "gint", cuda_graphs: Optional[bool] = None):
    """
    Register the gint backend with torch.compile.

    Args:
        name: The name to register the backend under (default: "gint")
        cuda_graphs: If True (or env ``GINT_CUDA_GRAPHS=1``), wrap the
            compiled forward in ``torch.cuda.make_graphed_callables`` so
            subsequent calls replay a CUDA graph. Default: False.

    Usage:
        >>> from gint.conductor import register_backend
        >>> register_backend(cuda_graphs=True)
        >>>
        >>> @torch.compile(backend="gint")
        >>> def my_function(x, y):
        >>>     return x + y
    """
    use_graphs = _resolve_cuda_graphs(cuda_graphs)
    backend_fn = _make_gint_backend(cuda_graphs=use_graphs)
    try:
        torch._dynamo.register_backend(name=name, compiler_fn=backend_fn)
        suffix = " (cuda_graphs=True)" if use_graphs else ""
        print(f"Successfully registered gint backend as '{name}'{suffix}")
    except AttributeError:
        raise RuntimeError(
            "torch._dynamo not available. "
            "Please ensure you have PyTorch 2.0+ installed with dynamo support."
        )
