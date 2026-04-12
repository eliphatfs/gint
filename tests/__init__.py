import os
import shutil
import unittest


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _llvm_link_available():
    return shutil.which('llvm-link') is not None


def _is_cuda_backend():
    backend = os.environ.get('GINT_BACKEND', '').lower()
    if backend == 'hip':
        return False
    if backend == 'cuda':
        return True
    # Auto-detect: try cuda-bindings import
    try:
        import cuda.bindings.driver  # noqa: F401
        return True
    except ImportError:
        return False


requires_gpu = unittest.skipUnless(_gpu_available(), "No GPU available")
requires_llvm_link = unittest.skipUnless(_llvm_link_available(), "llvm-link not found in PATH")
requires_cuda_backend = unittest.skipUnless(_is_cuda_backend(), "CUDA backend not available")
