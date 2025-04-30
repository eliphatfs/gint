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
])
TensorAddrs = ir.LiteralStructType([ir.VectorType(p_i8g, 8)])


class HTensorInfo(ctypes.Structure):
    _fields_ = [
        ("b_stride", ctypes.c_int32 * 4),
        ("b_size", ctypes.c_int32 * 4),
        ("t_stride", ctypes.c_int32),
        ("t_size", ctypes.c_int32),
        ("elm_size", ctypes.c_int32),
        ("resv_0", ctypes.c_int32),
    ]


class HTensorAddrs(ctypes.Structure):
    _fields_ = [
        ("addrs", ctypes.c_int64 * 8)
    ]
