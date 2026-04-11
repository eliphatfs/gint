import unittest


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_gpu = unittest.skipUnless(_gpu_available(), "No GPU available")
