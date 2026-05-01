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


def _resolve_options(kwargs, defaults):
    """Merge torch.compile ``mode`` and ``options`` into *defaults*.

    Returns ``(cuda_graphs, num_warmup_iters)``.

    Resolution order: registration-time defaults < ``mode`` < ``options``.
    """
    cuda_graphs = defaults[0]
    num_warmup_iters = defaults[1]

    mode = kwargs.get("mode", None)
    if mode == "no-cuda-graph":
        cuda_graphs = False

    opts = kwargs.get("options", None)
    if opts is not None:
        cuda_graphs = opts.get("cuda_graphs", cuda_graphs)
        num_warmup_iters = opts.get("num_warmup_iters", num_warmup_iters)
    return cuda_graphs, num_warmup_iters


def _make_gint_backend(cuda_graphs: bool, num_warmup_iters: int) -> Callable:
    """Return a torch.compile backend callable with the given defaults.

    The returned function accepts ``**kwargs`` so that ``torch.compile(backend="gint",
    options={...})`` can override the defaults at compile time via TorchDynamo's
    ``_TorchCompileWrapper``, which forwards ``mode`` and ``options`` as kwargs.
    """
    def gint_backend_fn(gm: GraphModule, example_inputs: List[torch.Tensor],
                        **kwargs) -> Callable:
        from torch._functorch.aot_autograd import aot_module_simplified

        _cuda_graphs, _num_warmup_iters = _resolve_options(
            kwargs, (cuda_graphs, num_warmup_iters))

        def compiler_fn(gm: GraphModule, example_inputs: List[torch.Tensor]):
            compiler = GintCompiler(gm, example_inputs)
            return compiler.compile()

        # aot_module_simplified is the lighter wrapper used by Inductor —
        # it skips the nn.Module-style trampolines that aot_module builds.
        # `inference_compiler` is the fast path taken under no_grad /
        # inference_mode; we use the same compiler for both since gint
        # doesn't do training.
        compiled = aot_module_simplified(
            gm, example_inputs,
            fw_compiler=compiler_fn,
            inference_compiler=compiler_fn,
        )

        if not _cuda_graphs:
            return compiled

        if not (torch.cuda.is_available() and all(
            isinstance(x, torch.Tensor) and x.is_cuda for x in example_inputs
        )):
            return compiled

        try:
            return torch.cuda.make_graphed_callables(
                compiled, tuple(example_inputs), num_warmup_iters=_num_warmup_iters
            )
        except Exception as e:
            print(f"[gint] CUDA graph capture failed ({e!r}); falling back to non-graphed path")
            return compiled

    return gint_backend_fn


def gint_backend(*, cuda_graphs: bool = True, num_warmup_iters: int = 1) -> Callable:
    """Return a gint backend callable for ``torch.compile``.

    The returned callable can be passed directly as the ``backend`` argument::

        @torch.compile(backend=gint_backend(cuda_graphs=False, num_warmup_iters=5))
        def fn(x, y):
            return x + y

    It also accepts ``options`` at compile time (these override the callable's
    defaults)::

        @torch.compile(backend=gint_backend(),
                       options={"cuda_graphs": False, "num_warmup_iters": 3})
        def fn(x, y):
            return x + y

    For the simple default case, ``backend="gint"`` is equivalent::

        @torch.compile(backend="gint")
        def fn(x, y):
            return x + y

    Args:
        cuda_graphs: If True, wrap the compiled callable with
            ``torch.cuda.make_graphed_callables`` so subsequent calls
            replay a CUDA graph.
        num_warmup_iters: Iterations the side-stream warmup runs before
            capture. Default 1 — sufficient to populate gint's per-shape
            device buffer cache. Don't use 0: allocations would land
            inside the captured region and break capture.
    """
    return _make_gint_backend(cuda_graphs=cuda_graphs, num_warmup_iters=num_warmup_iters)


def register_backend(name: str, cuda_graphs: bool, num_warmup_iters: int = 1):
    """Register a gint torch.compile backend under ``name``.

    On ``import gint.conductor`` the package auto-registers:

    - ``"gint"`` — cuda_graphs=True (default).
    - ``"gint-no-cuda-graph"`` — legacy alias for cuda_graphs=False.

    Non-default options can be set via ``torch.compile``'s ``options`` dict::

        @torch.compile(backend="gint",
                       options={"cuda_graphs": False, "num_warmup_iters": 5})

    Prefer ``gint_backend(cuda_graphs=..., num_warmup_iters=...)`` for defaults
    that differ from the registered ``"gint"`` backend.

    Args:
        name: The name to register the backend under.
        cuda_graphs: If True, wrap the compiled forward in
            ``torch.cuda.make_graphed_callables`` so subsequent calls
            replay a CUDA graph.
        num_warmup_iters: Iterations the side-stream warmup runs before
            capture. Default 1 — sufficient to populate gint's per-shape
            device buffer cache. Don't use 0: allocations would land
            inside the captured region and break capture.
    """
    backend_fn = _make_gint_backend(cuda_graphs=cuda_graphs, num_warmup_iters=num_warmup_iters)
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