import numpy
import unittest
import cuda.bindings.driver as cuda
from llvmlite import ir
from typing import Callable, Iterable
from gint.kernel.platforms.common import *
from gint.scripts.gen_llir import invoke_clang_shim
from gint.kernel.platforms.nvptx import NVPTXIRBuilder
from gint.host.cuda.driver import DriverContext, ptx_link, launch_kernel, check_cuda_error


class TestGenIRIntrinsicsNVPTX(unittest.TestCase):
    
    def test_trivial(self):
        LL = NVPTXIRBuilder.create_kernel_module(ir.FunctionType(void, []), "trivial")
        LL.ret_void()
        
        ptx = invoke_clang_shim(LL.emit())
        
        with DriverContext(0) as dctx:
            kernel = ptx_link(dctx, ptx, b"trivial")
            launch_kernel(kernel, grid_dim=1, block_dim=1, sync=True)
    
    def test_printf(self):
        LL = NVPTXIRBuilder.create_kernel_module(ir.FunctionType(void, [i32]), "test_printf")
        LL.printf("Hello World %d, ", LL.arg(0))
        LL.printf("Hello World %d %c! %s\n", LL.arg(0), i8(65), LL.string_literal("mooo"))
        LL.ret_void()
        
        ptx = invoke_clang_shim(LL.emit())
        
        with DriverContext(0) as dctx:
            kernel = ptx_link(dctx, ptx, b"test_printf")
            launch_kernel(kernel, 42, grid_dim=1, block_dim=1, sync=True)

    def test_sreg_bidx(self):
        expected = numpy.zeros([4, 64], dtype=numpy.int32)
        for i in range(4):
            expected[i] = i
        self._helper_test_sreg(expected.reshape(-1).tolist(), lambda LL: LL.block_idx_x())

    def test_sreg_tidx(self):
        expected = numpy.zeros([4, 64], dtype=numpy.int32)
        for i in range(64):
            expected[:, i] = i
        self._helper_test_sreg(expected.reshape(-1).tolist(), lambda LL: LL.thread_idx_x())

    def test_sreg_warpsize(self):
        expected = numpy.zeros([4, 64], dtype=numpy.int32)
        expected[:] = 32
        self._helper_test_sreg(expected.reshape(-1).tolist(), lambda LL: LL.warp_size())

    def test_sreg_laneid(self):
        expected = numpy.zeros([4, 64], dtype=numpy.int32)
        for i in range(32):
            expected[:, i] = i
            expected[:, i + 32] = i
        self._helper_test_sreg(expected.reshape(-1).tolist(), lambda LL: LL.lane_id())
    
    def test_warp_allreduce_sum(self):
        self._helper_test_warp_allreduce(sum, EReducePrimitiveOp.Sum)
    
    def test_warp_allreduce_max(self):
        self._helper_test_warp_allreduce(max, EReducePrimitiveOp.Max)
    
    def test_warp_allreduce_min(self):
        self._helper_test_warp_allreduce(min, EReducePrimitiveOp.Min)

    def _helper_test_warp_allreduce(self, py_red: Callable[[Iterable[int]], int], op: EReducePrimitiveOp):
        expected = numpy.zeros([4, 64], dtype=numpy.int32)
        for i in range(4):
            warp1 = py_red(i + j for j in range(32))
            warp2 = py_red(i + j for j in range(32, 64))
            expected[i, :32] = warp1
            expected[i, 32:] = warp2
        self._helper_test_sreg(
            expected.reshape(-1).tolist(),
            lambda LL: LL.fptosi(LL.warp_allreduce_f32(LL.sitofp(LL.add(LL.block_idx_x(), LL.thread_idx_x()), f32), op), i32)
        )

    def _helper_test_sreg(self, expected: list[int], get_fn: Callable[[NVPTXIRBuilder], ir.Value]):
        LL = NVPTXIRBuilder.create_kernel_module(ir.FunctionType(void, [i32.as_pointer()]), "test_sreg")
        global_tid = LL.add(LL.mul(LL.block_idx_x(), i32(64)), LL.thread_idx_x())
        LL.store(get_fn(LL), LL.gep(LL.arg(0), [global_tid]))
        LL.ret_void()
        
        ptx = invoke_clang_shim(LL.emit())
        
        with DriverContext(0) as dctx:
            kernel = ptx_link(dctx, ptx, b"test_sreg")
            err, dptr = cuda.cuMemAlloc(4 * 64 * 4)
            check_cuda_error(err)
            launch_kernel(kernel, dptr, grid_dim=4, block_dim=64, sync=True)
            host_buffer = numpy.zeros([4 * 64], dtype=numpy.int32)
            cuda.cuMemcpyDtoH(host_buffer, dptr, 4 * 64 * 4)
            self.assertListEqual(host_buffer.tolist(), expected)
