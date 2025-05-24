import enum
from llvmlite import ir
from llvmlite.ir.types import _BaseFloatType


class EReducePrimitiveOp(enum.Enum):
    Sum = 1
    Max = 2
    Min = 3
    Prod = 4


class BFloat16Type(_BaseFloatType):
    """
    The type for single-precision floats.
    """
    null = '0.0'
    intrinsic_name = 'bf16'

    def __str__(self):
        return 'bfloat'

    def format_constant(self, value):
        raise ValueError("BF16 constants not supported. Use conversion from FP32 instead!")


BFloat16Type._create_instance()


i1 = ir.IntType(1)  # bool
i8 = ir.IntType(8)
i32 = ir.IntType(32)
i64 = ir.IntType(64)
f16 = ir.HalfType()
f32 = ir.FloatType()
f64 = ir.DoubleType()
bf16 = BFloat16Type()
void = ir.VoidType()

p_i8 = i8.as_pointer()
p_i8g = i8.as_pointer(1)
