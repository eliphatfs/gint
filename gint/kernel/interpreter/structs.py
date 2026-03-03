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
    ir.ArrayType(i16, MAX_N_TENSORS),  # constraint 1 width weight
    ir.ArrayType(i16, MAX_N_TENSORS),  # constraint 1 thread weight
    ir.ArrayType(i16, MAX_N_TENSORS),  # constraint 1 offset weight
    ir.ArrayType(i16, MAX_N_TENSORS),  # constraint 2 width weight
    ir.ArrayType(i16, MAX_N_TENSORS),  # constraint 2 thread weight
    ir.ArrayType(i16, MAX_N_TENSORS),  # constraint 2 offset weight
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
        ("c1_ww", ctypes.c_uint16 * MAX_N_TENSORS),
        ("c1_wt", ctypes.c_uint16 * MAX_N_TENSORS),
        ("c1_wo", ctypes.c_uint16 * MAX_N_TENSORS),
        ("c2_ww", ctypes.c_uint16 * MAX_N_TENSORS),
        ("c2_wt", ctypes.c_uint16 * MAX_N_TENSORS),
        ("c2_wo", ctypes.c_uint16 * MAX_N_TENSORS),
    ]
