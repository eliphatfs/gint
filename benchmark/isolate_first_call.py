"""Measure fresh first-call cost of one backend in isolation.

Each subprocess invocation runs prime_cuda(), builds a torch.compile callable
for a single backend, and times the first call. This makes the fatbin load
(cuModuleLoadData on ~25 MB compressed) attributable to whichever backend
triggers it first in real usage — instead of being hidden by an earlier impl
in the bench loop.

Usage:
    python benchmark/isolate_first_call.py gint
    python benchmark/isolate_first_call.py gint-no-cuda-graph
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gint.conductor  # noqa: F401


def add3(a, b, c):
    return a + b + c


def prime_cuda():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    for _ in range(3):
        z = x + y + x
        del z
    torch.cuda.synchronize()


def main(backend):
    prime_cuda()
    N = 1 << 24
    torch.manual_seed(0)
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.randn(N, device='cuda', dtype=torch.float32)

    if os.environ.get('GINT_PRIME', '0') == '1':
        # Force gint fatbin load + first-launch cost to be paid before timing.
        torch.cuda.synchronize()
        ts = time.perf_counter()
        primer = torch.compile(add3, backend='gint-no-cuda-graph')
        primer(
            torch.randn(64, device='cuda'),
            torch.randn(64, device='cuda'),
            torch.randn(64, device='cuda'),
        )
        torch.cuda.synchronize()
        print(f"  gint prime cost: {(time.perf_counter() - ts) * 1000:.2f} ms")

    compiled = torch.compile(add3, backend=backend)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = compiled(a, b, c)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ref = a + b + c
    assert torch.allclose(ref, out, atol=1e-5)

    print(f"{backend:<24} first-call: {(t1 - t0) * 1000:.2f} ms")


if __name__ == '__main__':
    main(sys.argv[1])
