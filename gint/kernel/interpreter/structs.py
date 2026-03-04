import ctypes
from llvmlite import ir
from ..platforms.common import *


TensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, MAX_N_TENSORS),  # tensor base, need reinterpret cast (llir bitcast) before loading
    ir.ArrayType(i32, MAX_N_TENSORS),  # element size in bytes

    ir.ArrayType(ir.VectorType(i32, 4), MAX_N_TENSORS),  # batch strides
    ir.ArrayType(ir.VectorType(i32, 4), MAX_N_TENSORS),  # batch shape
    ir.ArrayType(ir.VectorType(i32, 2), MAX_N_TENSORS),  # shape dim/block stride 1
    ir.ArrayType(ir.VectorType(i32, 2), MAX_N_TENSORS),  # shape dim/block stride 2 (2d only)
    ir.ArrayType(ir.VectorType(i32, 2), MAX_N_TENSORS),  # block grid dim 1 and 2
    ir.ArrayType(ir.VectorType(i32, 2), MAX_N_TENSORS),  # block grid step 1 and 2
])


class HTensorInfo(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_int64 * MAX_N_TENSORS),
        ("elm_size", ctypes.c_int32 * MAX_N_TENSORS),

        ("batch_strides", (ctypes.c_int32 * 4) * MAX_N_TENSORS),
        ("batch_shape", (ctypes.c_int32 * 4) * MAX_N_TENSORS),
        ("block_shape_stride_1", (ctypes.c_int32 * 2) * MAX_N_TENSORS),
        ("block_shape_stride_2", (ctypes.c_int32 * 2) * MAX_N_TENSORS),
        ("block_grid_dims", (ctypes.c_int32 * 2) * MAX_N_TENSORS),
        ("block_grid_steps", (ctypes.c_int32 * 2) * MAX_N_TENSORS),
    ]
