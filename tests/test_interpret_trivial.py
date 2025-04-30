import numpy
import ctypes
import unittest
from cuda import cuda
from gint.scripts.gen_llir import invoke_clang_shim
from gint.kernel.interpreter.main import build_interpreter_main_nvptx
from gint.scripts.driver import DriverContext, ptx_link, launch_kernel, check_cuda_error


class TestInterpretTrivial(unittest.TestCase):
    
    def test_halt(self):
        mod = build_interpreter_main_nvptx()
        ptx = invoke_clang_shim(mod.emit())
        with DriverContext(0) as dctx:
            func = ptx_link(dctx, ptx, b'geval')
            err, dptr = cuda.cuMemAlloc(64)
            check_cuda_error(err)
            code = numpy.array([0, 0], dtype=numpy.int32)
            cuda.cuMemcpyHtoD(dptr, code, 8)
            launch_kernel(
                func,
                dptr, ctypes.c_void_p(0), 0,
                grid_dim=1, block_dim=32, sync=True
            )
            cuda.cuMemFree(dptr)
