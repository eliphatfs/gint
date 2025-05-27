from llvmlite import ir
from dataclasses import dataclass
from ..platforms.common import *


@dataclass
class RegisterSetSpec:
    name: str
    base: int
    num: int
    ty: ir.Type
    init: ir.Constant


@dataclass
class InterpreterStateSpec:
    ilp: int
    rf0: RegisterSetSpec
    rf1: RegisterSetSpec
    rf2: RegisterSetSpec
    rf3: RegisterSetSpec
    rss: RegisterSetSpec
    rof: RegisterSetSpec
    
    def __post_init__(self):
        self.flat_reg_inits()  # check bases are right
    
    def flat_reg_inits(self) -> list[tuple[str, ir.Constant]]:
        sets = [self.rf0, self.rf1, self.rf2, self.rf3, self.rss, self.rof]
        flat = []
        for s in sets:
            assert s.base == len(flat), s.name
            for i in range(s.num):
                flat.append((s.name + '_' + str(i), s.init))
        return flat


def get_spec(ilp: int = 4) -> InterpreterStateSpec:
    flat_rspec = [
        RegisterSetSpec('rf0', ilp * 0, ilp, f32, f32(0.0)),
        RegisterSetSpec('rf1', ilp * 1, ilp, f32, f32(ir.Undefined)),
        RegisterSetSpec('rf2', ilp * 2, ilp, f32, f32(ir.Undefined)),
        RegisterSetSpec('rf3', ilp * 3, ilp, f32, f32(ir.Undefined)),
        RegisterSetSpec('rss', ilp * 4 + 0, 5, i32, i32(ir.Undefined)),  # ilp and warp stride and limit; ilp thread offset contribution stride
        RegisterSetSpec('rof', ilp * 4 + 5, 1, p_i8g, p_i8g(ir.Undefined)),  # block offset
    ]
    return InterpreterStateSpec(
        ilp=ilp,
        **{v.name: v for v in flat_rspec}
    )


class RegisterSet(object):
    
    def __init__(self, flat_regs: list, spec: RegisterSetSpec) -> None:
        self.flat_regs = flat_regs
        self.base = spec.base
        self.num = spec.num
    
    def __getitem__(self, index: int) -> ir.Value:
        if index >= self.num:
            raise IndexError("Register set index out of range", index)
        return self.flat_regs[index + self.base]

    def __setitem__(self, index: int, val: ir.Value):
        self.flat_regs[index + self.base] = val


class InterpreterState:
    
    def __init__(self, regs: list[ir.Value], operand: ir.Value, spec: InterpreterStateSpec) -> None:
        super().__init__()
        self.assn_regs = regs.copy()
        self.operand = operand
        self.spec = spec
    
    def __getitem__(self, regset_spec: RegisterSetSpec) -> RegisterSet:
        return RegisterSet(self.assn_regs, regset_spec)

    def __setitem__(self, regset_spec: RegisterSetSpec, val_list: list[ir.Value]):
        assert len(val_list) == regset_spec.num
        for i in range(regset_spec.num):
            self[regset_spec][i] = val_list[i]
