from llvmlite import ir
from ..platforms.common import *


TensorInfo = ir.LiteralStructType([
    ir.VectorType(i32, 4),  # block strides, lowest dim share with ilp
    ir.VectorType(i32, 4),  # block shapes, lowest dim div by ilp
    i32,  # thread dim stride
    i32,  # thread dim shape
    i32,  # element size in bytes
    i32,  # reserved
    p_i8g,  # tensor base, need reinterpret cast (llir bitcast) before loading
])
