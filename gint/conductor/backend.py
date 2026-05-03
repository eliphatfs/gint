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


def _has_symint_shape(t) -> bool:
    """True if *t* is a tensor with at least one SymInt-typed dim."""
    if not isinstance(t, torch.Tensor):
        return False
    for d in t.shape:
        if isinstance(d, torch.SymInt):
            return True
    return False


def _resolve_options(kwargs, defaults):
    """Merge torch.compile ``mode`` and ``options`` into *defaults*.

    Returns ``(cuda_graphs, num_warmup_iters, clone_outputs)``.

    Resolution order: registration-time defaults < ``mode`` < ``options``.
    """
    cuda_graphs, num_warmup_iters, clone_outputs = defaults

    mode = kwargs.get("mode", None)
    if mode == "no-cuda-graph":
        cuda_graphs = False

    opts = kwargs.get("options", None)
    if opts is not None:
        cuda_graphs = opts.get("cuda_graphs", cuda_graphs)
        num_warmup_iters = opts.get("num_warmup_iters", num_warmup_iters)
        clone_outputs = opts.get("clone_outputs", clone_outputs)
    return cuda_graphs, num_warmup_iters, clone_outputs


def _make_gint_backend(cuda_graphs: bool, num_warmup_iters: int,
                       clone_outputs: bool = True) -> Callable:
    """Return a torch.compile backend callable with the given defaults.

    The returned function accepts ``**kwargs`` so that ``torch.compile(backend="gint",
    options={...})`` can override the defaults at compile time via TorchDynamo's
    ``_TorchCompileWrapper``, which forwards ``mode`` and ``options`` as kwargs.
    """
    def gint_backend_fn(gm: GraphModule, example_inputs: List[torch.Tensor],
                        **kwargs) -> Callable:
        from torch._functorch.aot_autograd import aot_module_simplified

        _cuda_graphs, _num_warmup_iters, _clone_outputs = _resolve_options(
            kwargs, (cuda_graphs, num_warmup_iters, clone_outputs))

        # Closure: track the gint subgraph count for each frame compile so
        # the outer backend (below) can skip cuda-graph capture for frames
        # the gint compiler had no work in. ``aot_module_simplified`` may
        # call ``compiler_fn`` more than once per backend call (fwd+bwd
        # for training, or partition decompositions); we reset per
        # backend call below.
        compile_stats = {"num_gint_subgraphs": 0}

        def compiler_fn(gm: GraphModule, example_inputs: List[torch.Tensor]):
            # Fail fast on SymInt-shaped inputs. Dynamo's automatic
            # dynamic-shape promotion is disabled at registration time
            # (see ``register_backend``), so this only fires when the
            # user explicitly opts in via ``dynamic=True`` or exceeds
            # ``cache_size_limit``. SymInts here would propagate into
            # ``ProgramTensorInfo`` / ``grid_dim`` / output allocation
            # which all need concrete ints.
            if any(_has_symint_shape(t) for t in example_inputs):
                raise RuntimeError(
                    "gint backend received SymInt-shaped inputs (dynamic "
                    "shapes), which it does not support. Use "
                    "``gint.conductor.compile(fn)`` (drop-in for "
                    "``torch.compile``; scopes dynamo's "
                    "automatic_dynamic_shapes=False patch to the wrapped "
                    "call only), or pass ``dynamic=False`` to "
                    "``torch.compile``, or call "
                    "``torch._dynamo.mark_static(t, dim)`` on the dynamic dim."
                )
            import os as __os
            if __os.environ.get('GINT_CG_DUMP_AOT_INT64'):
                # Dump any AOT graph that has an int64 input.
                ph = [n for n in gm.graph.nodes if n.op == 'placeholder']
                vals = [n.meta.get('val') for n in ph]
                dtypes = [getattr(v, 'dtype', None) for v in vals]
                if any(d == torch.int64 for d in dtypes):
                    print(f"=== AOT FX with int64 input (dtypes={dtypes}) ===")
                    print(gm.graph)
                    print("=== end ===")
            compiler = GintCompiler(gm, example_inputs)
            executor = compiler.compile()
            compile_stats["num_gint_subgraphs"] += getattr(
                executor, "num_gint_subgraphs", 0)
            return executor

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

        # Subgraphs with no tensor inputs can't be meaningfully captured
        # (``torch.cuda.graph`` then warns "The CUDA Graph is empty"
        # because there's no work on the captured stream). Pass through.
        if len(example_inputs) == 0:
            return compiled

        # Frames with no gint subgraphs are pure eager fallback. Wrapping
        # adds no gint benefit, and the cost (capture + replay overhead)
        # would just slow them down. Skip.
        if compile_stats["num_gint_subgraphs"] == 0:
            return compiled
        # Skip frames with non-fp32 inputs. These are typically AOT
        # subgraphs containing int64 advanced indexing (``aten.index.Tensor``
        # with int64 index tensors) — the rasterization-style pipelines
        # in diffrp emit them via ``torch.gather`` over per-pixel triangle
        # IDs. When we capture such a frame, intermediate tensors that
        # the eager fallback allocates inside the captured pool end up
        # aliasing addresses that another captured graph's replay later
        # writes to, silently corrupting outputs the user has already
        # received from this frame. The repro path is hard to isolate
        # outside the full diffrp pipeline (small synthetic multi-frame
        # tests don't reproduce it) and looks like an interaction
        # between AOT's mixed eager/gint graph and torch's CUDA-graph
        # mempool tracking; the dtype heuristic here is the conservative
        # fix that keeps the rest of the frames graphed and gives the
        # fish render correct output at ~80% of eager fps. Override
        # with ``GINT_CG_SKIP_INT=0`` if you know your AOT graphs are
        # capture-safe (e.g. only fp32 pointwise pipelines).
        import os as _os
        if _os.environ.get('GINT_CG_SKIP_INT', '1') == '1' and any(
                getattr(t, 'dtype', None) != torch.float32
                for t in example_inputs):
            return compiled
        try:
            return _direct_cudagraph_wrap(compiled, tuple(example_inputs),
                                          clone_outputs=_clone_outputs)
        except Exception as e:
            print(f"[gint] CUDA graph capture failed ({e!r}); falling back to non-graphed path")
            return compiled

    return gint_backend_fn


def _direct_cudagraph_wrap(compiled, sample_args, clone_outputs: bool = True):
    """Inference-only raw ``torch.cuda.CUDAGraph`` wrapper. Warmup on a
    side stream, capture once on the main stream into a fresh mempool,
    and replay on each call.

    *clone_outputs* (default True): clone replayed outputs out of the
    graph pool before returning. Required when the wrapped callable's
    output may be HELD by the caller across other captured-graph
    replays in the same process — without it, those replays can land
    on the same pool addresses and silently corrupt the held tensor.
    The diffrp fish render hits this: ``gint_compile`` wraps a function
    with many graph breaks (each AOT frame becomes its own captured
    graph), and per-frame outputs flow forward as inputs to later
    frames' captures. Inductor's ``cudagraph_trees`` raises an explicit
    "accessing tensor output of CUDAGraphs that has been overwritten
    by a subsequent run" for the same pattern.

    Set False when the wrapped function is the only torch.compile'd
    thing in the program AND outputs are consumed before the next call
    (e.g. a single-kernel benchmark). Eliminates one DtoD memcpy per
    output per call — for memory-bound kernels with large outputs the
    saving is significant (add3: ~150 µs/call on 16M fp32 outputs).
    Matches ``torch.cuda.make_graphed_callables``'s historical
    detach-only behavior.

    No autograd handling — gint is inference-only and ``compile()``'s
    ``torch.no_grad()`` scope (or the user's own) keeps grad off."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        compiled(*sample_args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(g, pool=pool):
        raw_outputs = compiled(*sample_args)
    # Preserve the original output container (single Tensor, list, tuple)
    # so callers (notably AOT autograd's ``runtime_wrapper``, which
    # expects a list/tuple) see the same structure.
    is_tensor = isinstance(raw_outputs, torch.Tensor)
    static_outputs = (raw_outputs,) if is_tensor else tuple(raw_outputs)
    out_kind = type(raw_outputs)
    _post = (lambda o: o.clone()) if clone_outputs else (lambda o: o.detach())

    def wrapper(*args):
        for s_in, a in zip(sample_args, args):
            if s_in.data_ptr() != a.data_ptr():
                s_in.copy_(a)
        g.replay()
        cloned = tuple(_post(o) for o in static_outputs)
        if is_tensor:
            return cloned[0]
        return out_kind(cloned)

    wrapper._gint_keepalive = (g, pool, sample_args, static_outputs)
    return wrapper


def compile(fn=None, *, backend: str = "gint", **kwargs) -> Callable:
    """``torch.compile`` drop-in that scopes dynamo to gint's static-shape model.

    gint bakes shapes/strides/grid_dim into per-shape bytecode, so it can't
    compile FX graphs with SymInt-shaped FakeTensors. Dynamo's
    ``automatic_dynamic_shapes`` would normally promote a dim to dynamic on
    the second recompile, breaking us. This wrapper applies
    ``torch._dynamo.config.patch(automatic_dynamic_shapes=False,
    assume_static_by_default=True)`` only while the wrapped callable is
    executing — so dynamo's tracing (which happens inside the call) sees
    the patched config, but other backends in the same process keep their
    original config.

    Use this instead of ``torch.compile(backend="gint")`` when you want
    per-shape recompilation without flipping global dynamo config::

        from gint.conductor import compile as gint_compile

        @gint_compile
        def fn(a, b):
            return a + b

    Pass ``backend=gint_backend(...)`` (or any other gint backend name) to
    customise CUDA-graph behavior::

        @gint_compile(backend=gint_backend(cuda_graphs=False))
        def fn(a, b): ...
    """
    import functools

    if fn is None:
        return functools.partial(compile, backend=backend, **kwargs)

    base = torch.compile(fn, backend=backend, **kwargs)

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False,
            assume_static_by_default=True,
        ):
            return base(*args, **kw)

    return wrapper


def gint_backend(*, cuda_graphs: bool = True, num_warmup_iters: int = 1,
                 clone_outputs: bool = True) -> Callable:
    """Return a gint backend callable for ``torch.compile``.

    The returned callable can be passed directly as the ``backend`` argument::

        @torch.compile(backend=gint_backend(cuda_graphs=False, num_warmup_iters=5))
        def fn(x, y):
            return x + y

    It also accepts ``options`` at compile time (these override the callable's
    defaults)::

        @torch.compile(backend=gint_backend(),
                       options={"cuda_graphs": False, "num_warmup_iters": 3,
                                "clone_outputs": False})
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
        clone_outputs: If True (default, safe), each replay's outputs are
            cloned out of the captured graph pool before being returned to
            the caller. Required when outputs may be HELD across other
            captured-graph replays in the same process (e.g. multi-frame
            ``gint_compile`` of a function with graph breaks where each
            frame's output flows forward as another frame's input). Set
            False when the wrapped function is the only torch.compile'd
            thing AND outputs are consumed before the next call (typical
            single-kernel benchmark / single-stage inference). Saves one
            DtoD memcpy per output per call — significant for memory-bound
            kernels with large outputs.
    """
    return _make_gint_backend(cuda_graphs=cuda_graphs,
                              num_warmup_iters=num_warmup_iters,
                              clone_outputs=clone_outputs)


def register_backend(name: str, cuda_graphs: bool, num_warmup_iters: int = 1,
                     clone_outputs: bool = True):
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
    backend_fn = _make_gint_backend(cuda_graphs=cuda_graphs,
                                    num_warmup_iters=num_warmup_iters,
                                    clone_outputs=clone_outputs)
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