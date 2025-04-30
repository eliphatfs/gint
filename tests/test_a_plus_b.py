import numpy
import ctypes
import unittest
from cuda import cuda
from gint.scripts.gen_llir import invoke_clang_shim
from gint.kernel.interpreter.main import build_interpreter_main_nvptx, ILP
from gint.kernel.interpreter.structs import HTensorInfo
from gint.scripts.driver import DriverContext, ptx_link, launch_kernel, check_cuda_error


class BatchAddProgram:
    
    def __init__(self, intp: cuda.CUfunction, last_dim: int):
        self.last_dim = last_dim
        self.program = numpy.array([
            1, 0,  # load tensor infos
            2, 0,  # load global idx 0 tensor 0
            8, 0,  # mov f0 f1
            2, 1,  # load global idx 0 tensor 1
            4, 0,  # faddto
            3, 2,  # store global idx 0 tensor 2
            2, 16 * 32,  # load global idx 32 tensor 0
            8, 0,  # mov f0 f1
            2, 16 * 32 + 1,  # load global idx 32 tensor 1
            4, 0,  # faddto
            3, 16 * 32 + 2,  # store global idx 32 tensor 2
            0, 0,  # halt
        ], dtype=numpy.int32)
        err, dcode = cuda.cuMemAlloc(len(self.program) * 4)
        check_cuda_error(err)
        check_cuda_error(cuda.cuMemcpyHtoD(dcode, self.program, len(self.program) * 4))
        self.dcode = dcode
        self.intp = intp

    def run(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        assert a.dtype == numpy.float32
        assert b.dtype == numpy.float32
        B1, C = a.shape
        assert C <= self.last_dim, C
        B2, C2 = b.shape
        assert C == C2, [C, C2]
        assert B1 == B2, [B1, B2]
        _, da = check_cuda_error(cuda.cuMemAlloc(B1 * C * 4))
        _, db = check_cuda_error(cuda.cuMemAlloc(B2 * C * 4))
        _, dc = check_cuda_error(cuda.cuMemAlloc(B1 * C * 4))
        _, hinfo = cuda.cuMemAllocHost(ctypes.sizeof(HTensorInfo))
        ti = HTensorInfo.from_address(int(hinfo))
        _, dinfo = check_cuda_error(cuda.cuMemAlloc(ctypes.sizeof(ti)))
        c = numpy.empty([B1, C], dtype=numpy.float32)
        for i, (h, d) in enumerate(zip([a, b, c], [da, db, dc])):
            d: cuda.CUdeviceptr
            ti.b_stride[i][0] = h.strides[0] // 4
            ti.b_size[i][0] = h.shape[0]
            for j in range(1, 4):
                ti.b_stride[i][j] = 0
                ti.b_size[i][j] = 1
            ti.t_stride[i] = h.strides[1] // 4
            ti.t_size[i] = h.shape[1]
            ti.elm_size[i] = 4
            ti.base_ptr[i] = int(d)
        check_cuda_error(cuda.cuMemcpyHtoD(da, a, B1 * C * 4))
        check_cuda_error(cuda.cuMemcpyHtoD(db, b, B2 * C * 4))
        check_cuda_error(cuda.cuMemcpyHtoDAsync(dinfo, hinfo, ctypes.sizeof(ti), 0))
        check_cuda_error(launch_kernel(self.intp, self.dcode, dinfo, 3, grid_dim=(h.shape[0] + ILP - 1) // ILP, block_dim=32, sync=True))
        check_cuda_error(cuda.cuMemcpyDtoH(c, dc, B1 * C * 4))
        check_cuda_error(cuda.cuMemFree(da))
        check_cuda_error(cuda.cuMemFree(db))
        check_cuda_error(cuda.cuMemFree(dc))
        check_cuda_error(cuda.cuMemFree(dinfo))
        check_cuda_error(cuda.cuMemFreeHost(hinfo))
        return c


class TestInterpretAPB(unittest.TestCase):
    
    def test_a_plus_b(self):
        mod = build_interpreter_main_nvptx()
        ptx = invoke_clang_shim(mod.emit())
        with DriverContext(0) as dctx:
            func = ptx_link(dctx, ptx, b'geval')
            bap32 = BatchAddProgram(func, 64)
            for s in [1, 4, 6, 31, 32, 1000, 200000]:
                for p in [1, 16, 18, 32, 64]:
                    a = numpy.random.rand(s, p).astype(numpy.float32)
                    b = numpy.random.rand(s, p).astype(numpy.float32)
                    c_host = a + b
                    c_device = bap32.run(a, b)
                    numpy.testing.assert_allclose(c_device, c_host)
