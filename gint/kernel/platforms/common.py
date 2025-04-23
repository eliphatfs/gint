from llvmlite import ir


i1 = ir.IntType(1)  # bool
i8 = ir.IntType(8)
i32 = ir.IntType(32)
i64 = ir.IntType(64)
f16 = ir.HalfType()
f32 = ir.FloatType()
f64 = ir.DoubleType()
void = ir.VoidType()

p_i8 = i8.as_pointer()
