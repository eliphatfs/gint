"""
Torch.compile backend for gint.

This module provides integration with PyTorch's torch.compile infrastructure,
allowing gint to serve as a custom backend for compiling PyTorch models.

On import, two backends are registered automatically:

- ``"gint"`` — cuda_graphs=True, num_warmup_iters=1 (default).
- ``"gint-no-cuda-graph"`` — legacy alias for cuda_graphs=False.

Options can be passed via ``torch.compile``'s ``options`` dict::

    import gint.conductor  # noqa: F401  (registers backends)

    @torch.compile(backend="gint")
    def fn(x, y):
        return x + y

    @torch.compile(backend="gint",
                   options={"cuda_graphs": False, "num_warmup_iters": 5})
    def fn2(x, y):
        return x * y

Or via ``gint_backend()`` for baked-in defaults::

    from gint.conductor import gint_backend

    @torch.compile(backend=gint_backend(cuda_graphs=False))
    def fn3(x, y):
        return x * y
"""

from .backend import gint_backend, register_backend, compile

register_backend("gint", cuda_graphs=True)
register_backend("gint-no-cuda-graph", cuda_graphs=False)

__all__ = ['gint_backend', 'register_backend', 'compile']
