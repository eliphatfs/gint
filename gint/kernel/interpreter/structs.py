import ctypes
from llvmlite import ir
from ..platforms.common import *


TensorInfo = ir.LiteralStructType([
    ir.VectorType(i32, 4),  # block strides, lowest dim share with ilp
    ir.VectorType(i32, 4),  # block shapes, lowest dim div by ilp
    i32,  # thread dim stride
    i32,  # thread dim shape
    i32,  # element size in bytes
    i32,  # reserved_0
    i32,  # reserved_1
    i32,  # reserved_2
    p_i8g,  # tensor base, need reinterpret cast (llir bitcast) before loading
])


class HTensorInfo(ctypes.Structure):
    _fields_ = [
        ("b_stride", ctypes.c_int32 * 4),
        ("b_size", ctypes.c_int32 * 4),
        ("t_stride", ctypes.c_int32),
        ("t_size", ctypes.c_int32),
        ("elm_size", ctypes.c_int32),
        ("resv_0", ctypes.c_int32),
        ("resv_1", ctypes.c_int32),
        ("resv_2", ctypes.c_int32),
        ("base_ptr", ctypes.c_int64),
    ]
