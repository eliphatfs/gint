"""
Torch.compile backend for gint.

This module provides integration with PyTorch's torch.compile infrastructure,
allowing gint to serve as a custom backend for compiling PyTorch models.

On import, two backends are registered automatically:

- ``"gint"`` — default, with CUDA graph capture enabled.
- ``"gint-no-cuda-graph"`` — same backend without CUDA graph capture.

Usage::

    import gint.conductor  # noqa: F401  (registers backends)

    @torch.compile(backend="gint")
    def fn(x, y):
        return x + y
"""

from .backend import register_backend

register_backend("gint", cuda_graphs=True)
register_backend("gint-no-cuda-graph", cuda_graphs=False)

__all__ = ['register_backend']
