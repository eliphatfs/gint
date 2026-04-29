"""
Line-profile torch.cuda.make_graphed_callables for the gint add3 case.

Runs a single full first-call cycle of torch.compile(backend='gint'), with
make_graphed_callables instrumented by line_profiler. The compile is wrapped
in section timers as well so we can attribute startup latency between
dynamo/AOT compile vs. the graph-capture pipeline.
"""

import os
import sys
import time

import torch
from line_profiler import LineProfiler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gint.conductor  # noqa: F401  (registers backends)


def add3(a, b, c):
    return a + b + c


def prime_cuda():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    for _ in range(3):
        z = x + y + x
        del z
    torch.cuda.synchronize()


def main():
    prime_cuda()

    N = 1 << 24
    torch.manual_seed(0)
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.randn(N, device='cuda', dtype=torch.float32)

    lp = LineProfiler()
    lp.add_function(torch.cuda.make_graphed_callables)

    compiled = torch.compile(add3, backend='gint')

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lp.enable_by_count()
    out = compiled(a, b, c)
    lp.disable_by_count()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ref = a + b + c
    assert torch.allclose(ref, out, atol=1e-5)

    print(f"\nTotal first-call wall time: {(t1 - t0) * 1000:.2f} ms\n")
    lp.print_stats(output_unit=1e-3)  # report per-line time in ms


if __name__ == '__main__':
    main()
