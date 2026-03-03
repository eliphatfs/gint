import ctypes
from llvmlite import ir
from ..platforms.common import *


TensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, MAX_N_TENSORS),  # tensor base, need reinterpret cast (llir bitcast) before loading
    ir.ArrayType(i32, MAX_N_TENSORS),  # element size in bytes

    ir.ArrayType(ir.VectorType(i32, 4), MAX_N_TENSORS),  # block strides
    ir.ArrayType(ir.VectorType(i32, 4), MAX_N_TENSORS),  # block shapes
    ir.ArrayType(i32, MAX_N_TENSORS),  # block to thread stride (b2t)
    ir.ArrayType(i32, MAX_N_TENSORS),  # block to thread size (n_b2t)
    ir.ArrayType(i32, MAX_N_TENSORS),  # block to width stride (b2w)
    ir.ArrayType(i32, MAX_N_TENSORS),  # block to width size (n_b2w)
    ir.ArrayType(i32, MAX_N_TENSORS),  # thread stride
    ir.ArrayType(i32, MAX_N_TENSORS),  # width stride
    ir.ArrayType(i32, MAX_N_TENSORS),  # offset stride

    ir.ArrayType(i32, MAX_N_TENSORS),  # constraint 1 size
    ir.ArrayType(i32, MAX_N_TENSORS),  # constraint 2 size
    ir.ArrayType(ir.VectorType(i16, 4), MAX_N_TENSORS),  # constraint 1 width/thread/offset weight log2 packed
    ir.ArrayType(ir.VectorType(i16, 4), MAX_N_TENSORS),  # constraint 2 width/thread/offset weight log2 packed
])


class HTensorInfo(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_int64 * MAX_N_TENSORS),
        ("elm_size", ctypes.c_int32 * MAX_N_TENSORS),

        ("b_strides", (ctypes.c_int32 * 4) * MAX_N_TENSORS),
        ("b_sizes", (ctypes.c_int32 * 4) * MAX_N_TENSORS),
        ("b2t_stride", ctypes.c_int32 * MAX_N_TENSORS),
        ("b2t_size", ctypes.c_int32 * MAX_N_TENSORS),
        ("b2w_stride", ctypes.c_int32 * MAX_N_TENSORS),
        ("b2w_size", ctypes.c_int32 * MAX_N_TENSORS),
        ("t_stride", ctypes.c_int32 * MAX_N_TENSORS),
        ("w_stride", ctypes.c_int32 * MAX_N_TENSORS),
        ("o_stride", ctypes.c_int32 * MAX_N_TENSORS),

        ("c1_size", ctypes.c_int32 * MAX_N_TENSORS),
        ("c2_size", ctypes.c_int32 * MAX_N_TENSORS),
        ("c1_w", (ctypes.c_int16 * 4) * MAX_N_TENSORS),
        ("c2_w", (ctypes.c_int16 * 4) * MAX_N_TENSORS),
    ]
