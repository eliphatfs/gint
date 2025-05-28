import ctypes
from llvmlite import ir
from ..platforms.common import *


TensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, 8),  # tensor base, need reinterpret cast (llir bitcast) before loading
    ir.ArrayType(ir.VectorType(i32, 4), 8),  # block strides, lowest dim share with ilp
    ir.ArrayType(ir.VectorType(i32, 4), 8),  # block shapes, lowest dim div by ilp
    ir.ArrayType(ir.VectorType(i32, 4), 8),  # contribution strides of block to thread offset, lowest dim share with ilp
    ir.ArrayType(i32, 8),  # thread dim stride
    ir.ArrayType(i32, 8),  # thread dim shape
    ir.ArrayType(i32, 8),  # element size in bytes
])


class HTensorInfo(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_int64 * 8),
        ("b_stride", (ctypes.c_int32 * 4) * 8),
        ("b_size", (ctypes.c_int32 * 4) * 8),
        ("bt_ofs_stride", (ctypes.c_int32 * 4) * 8),
        ("t_stride", ctypes.c_int32 * 8),
        ("t_size", ctypes.c_int32 * 8),
        ("elm_size", ctypes.c_int32 * 8),
    ]
