"""Collect opcode frequencies from all @bytecode programs in the codebase.

Walks every SugarProgram, builds bytecode with dummy tensors, counts
opcode occurrences. Adds +1 to all registered opcodes so unobserved
ones are still included. Exports to gint/kernel/interpreter/opcode_frequencies.py.
"""

import collections
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gint.kernel.interpreter.main import INSNS
from gint.host.sugar import SugarProgram


class MockTensor:
    """Minimal TensorInterface-compatible object for frequency collection."""

    def __init__(self, shape, dtype=torch.float32):
        self.shape = tuple(shape)
        self._dtype = dtype
        dt = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        self.elm_size = dt.bits // 8
        self.typechr = {torch.float32: 'f', torch.float16: 'f', torch.bfloat16: 'f',
                        torch.float64: 'f', torch.int32: 'i', torch.int64: 'i',
                        torch.uint8: 'B', torch.int8: 'b'}.get(dtype, 'f')
        self.strides = tuple(reversed([1] + [s for s in shape[:-1]]))
        self.base_ptr = 0x1000
        self.ndim = len(self.shape)

    @property
    def typestr(self):
        return f'{self.typechr}{self.elm_size}'


def mock(shape, dtype=torch.float32):
    return MockTensor(shape, dtype)


def collect(prog, *args, **extra):
    """Invoke a SugarProgram's sugar_wrapper to record bytecode and return
    the flat opcode list (ignoring operands)."""
    bc, _tis, _tidx = prog.func(*args, REGW=4, WARP=32, **extra)
    return [op for op, _operand in bc]


def safe_collect(prog, name, *args, **extra):
    """Collect frequencies from a program, ignoring errors."""
    try:
        return collect(prog, *args, **extra)
    except Exception:
        print(f"  [warn] {name}: {traceback.format_exc()}")
        return []


def main():
    freq = collections.Counter()
    N = 256

    programs = [
        # test_frontend
        ("basic_expr1", lambda: __import__('tests.test_frontend', fromlist=['basic_expr1']).basic_expr1,
         (mock((N, 1)), mock((N, 1)), mock((N, 1))), {}),
        ("vector_expr2", lambda: __import__('tests.test_frontend', fromlist=['vector_expr2']).vector_expr2,
         (mock((N,)), mock((N,))), {'BLOCK': N}),
        ("meaningless_execute_everything", lambda: __import__('tests.test_frontend', fromlist=['meaningless_execute_everything']).meaningless_execute_everything,
         (mock((N,), torch.float32), mock((N,), torch.float16), mock((N,), torch.bfloat16), mock((N,), torch.uint8)), {}),
        ("indirect_arith_test", lambda: __import__('tests.test_frontend', fromlist=['indirect_arith_test']).indirect_arith_test,
         tuple(mock((N,)) for _ in range(7)), {}),
        ("packed_imm_test", lambda: __import__('tests.test_frontend', fromlist=['packed_imm_test']).packed_imm_test,
         (mock((N,)), mock((N,))), {}),
        # test_indirect
        ("vec_add", lambda: __import__('tests.test_indirect', fromlist=['vec_add']).vec_add,
         (mock((N,)), mock((N,)), mock((N,))), {}),
        ("vec_mul", lambda: __import__('tests.test_indirect', fromlist=['vec_mul']).vec_mul,
         (mock((N,)), mock((N,)), mock((N,))), {}),
        # test_cuda_graph_capture
        ("cg_vector_expr2", lambda: __import__('tests.test_cuda_graph_capture', fromlist=['vector_expr2']).vector_expr2,
         (mock((N,)), mock((N,))), {'BLOCK': N}),
        # test_rmsnorm
        ("rmsnorm", lambda: __import__('tests.test_rmsnorm', fromlist=['rmsnorm']).rmsnorm,
         (mock((2, 32, 64, 128)), mock((2, 32, 64, 128)), mock((128,))), {}),
        # test_bmm4x4
        ("bmm4x4_test", lambda: __import__('tests.test_bmm4x4', fromlist=['bmm4x4_kernel']).bmm4x4_kernel,
         (mock((1, 16)), mock((1, 16)), mock((1, 16))), {}),
        # test_inv4x4
        ("inv4x4_test", lambda: __import__('tests.test_inv4x4', fromlist=['inv4x4_kernel']).inv4x4_kernel,
         (mock((1, 16)), mock((1, 16))), {}),
        # matrix kernels
        ("bmm4x4", lambda: __import__('gint.host.matrix', fromlist=['bmm4x4_kernel']).bmm4x4_kernel,
         (mock((1, 16)), mock((1, 16)), mock((1, 16))), {}),
        ("inv4x4", lambda: __import__('gint.host.matrix', fromlist=['inv4x4_kernel']).inv4x4_kernel,
         (mock((1, 16)), mock((1, 16))), {}),
    ]

    for name, getter, args, extra in programs:
        try:
            prog = getter()
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        freq.update(safe_collect(prog, name, *args, **extra))

    # benchmark roofline -- collect for multiple degrees
    try:
        from benchmark.bench_roofline import _make_gint_poly
        for degree in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            prog = _make_gint_poly(degree)
            freq.update(safe_collect(prog, f"roofline_d{degree}",
                                     mock((N,)), mock((N,)), N=N, M=1))
    except Exception as e:
        print(f"  [skip] roofline: {e}")

    # benchmark lowlevel
    try:
        from benchmark.bench_lowlevel import (
            gint_add3_kernel, gint_rmsnorm_2d, gint_ggx_importance_kernel,
        )
        freq.update(safe_collect(gint_add3_kernel, "add3",
                                 mock((N,)), mock((N,)), mock((N,)), N=N, M=1))
        freq.update(safe_collect(gint_rmsnorm_2d, "rmsnorm_2d",
                                 mock((2, 32, 64, 128)), mock((2, 32, 64, 128)), mock((128,))))
        freq.update(safe_collect(gint_ggx_importance_kernel, "ggx_importance",
                                 mock((N, 2)), mock((N,)), mock((N, 2)), mock((N,)), N=N, M=1))
    except Exception as e:
        print(f"  [skip] lowlevel: {e}")

    # benchmark compile
    try:
        from benchmark.bench_compile import gint_rmsnorm_manual_fp32
        freq.update(safe_collect(gint_rmsnorm_manual_fp32, "rmsnorm_fp32",
                                 mock((2, 32, 64, 128)), mock((2, 32, 64, 128)), mock((128,))))
    except Exception as e:
        print(f"  [skip] compile: {e}")

    # -- Add +1 to all registered opcodes --
    for opcode in INSNS.values():
        freq[opcode] += 1

    # -- Export --
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'gint', 'kernel', 'interpreter', 'opcode_frequencies.py')
    with open(out_path, 'w') as f:
        f.write('# Auto-generated opcode frequencies for optimal dispatch tree\n')
        f.write('# Generated by scripts/collect_opcode_frequencies.py\n')
        f.write('FREQUENCIES = ')
        json.dump(dict(freq), f, indent=2)
        f.write('\n')

    print(f"Collected frequencies for {len(freq)} opcodes")
    print(f"Written to {out_path}")
    top = freq.most_common(20)
    for opcode, count in top:
        name = next((insn.__name__ for insn, oid in INSNS.items() if oid == opcode), f"op_{opcode}")
        print(f"  {name}: {count}")


if __name__ == '__main__':
    main()