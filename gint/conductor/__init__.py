"""
Torch.compile backend for gint.

This module provides integration with PyTorch's torch.compile infrastructure,
allowing gint to serve as a custom backend for compiling PyTorch models.
"""

from .backend import gint_backend, register_backend

__all__ = ['gint_backend', 'register_backend']
