"""
Core torch.compile backend implementation for gint.

This module implements the backend contract required by torch.compile,
converting FX graphs into gint bytecode programs.
"""

import torch
from typing import List, Callable, Optional
from torch.fx import GraphModule

from ..host.executor import TensorInterface, get_executor
from .compiler import GintCompiler


def gint_backend(gm: GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    Main entry point for the gint torch.compile backend.
    
    Uses AOTAutograd to functionalize the graph and then compiles it.
    """
    from torch._functorch.aot_autograd import aot_module
    
    def compiler_fn(gm: GraphModule, example_inputs: List[torch.Tensor]):
        compiler = GintCompiler(gm, example_inputs)
        return compiler.compile()
    
    # Use aot_module to functionalize and normalize the graph
    return aot_module(gm, fw_compiler=compiler_fn)


def register_backend(name: str = "gint"):
    """
    Register the gint backend with torch.compile.
    
    Args:
        name: The name to register the backend under (default: "gint")
        
    Usage:
        >>> from gint.conductor import register_backend
        >>> register_backend()
        >>> 
        >>> @torch.compile(backend="gint")
        >>> def my_function(x, y):
        >>>     return x + y
    """
    try:
        # Register with torch._dynamo
        torch._dynamo.register_backend(name=name, compiler_fn=gint_backend)
        print(f"Successfully registered gint backend as '{name}'")
    except AttributeError:
        raise RuntimeError(
            "torch._dynamo not available. "
            "Please ensure you have PyTorch 2.0+ installed with dynamo support."
        )
