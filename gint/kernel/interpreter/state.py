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
    rb0: RegisterSetSpec
    rofs: RegisterSetSpec
    
    def __post_init__(self):
        self.flat_reg_inits()  # check bases are right
    
    def flat_reg_inits(self) -> list[tuple[str, ir.Constant]]:
        sets = [self.rf0, self.rf1, self.rb0, self.rofs]
        flat = []
        for s in sets:
            assert s.base == len(flat)
            for i in range(s.num):
                flat.append((s.name + '_' + str(i), s.init))
        return flat


def get_spec(ilp: int = 4) -> InterpreterStateSpec:
    return InterpreterStateSpec(
        ilp=ilp,
        rf0=RegisterSetSpec('rf0', 0, ilp, f32, f32(0.0)),
        rf1=RegisterSetSpec('rf1', ilp, ilp, f32, f32(0.0)),
        rb0=RegisterSetSpec('rb0', ilp * 2, ilp, i1, i1(False)),
        rofs=RegisterSetSpec('rofs', ilp * 3, 1, i64, i64(0))
    )


class RegisterSet(object):
    
    def __init__(self, flat_regs: list, spec: RegisterSetSpec) -> None:
        self.flat_regs = flat_regs
        self.base = spec.base
    
    def __getitem__(self, index: int) -> ir.Value:
        return self.flat_regs[index + self.base]

    def __setitem__(self, index: int, val: ir.Value):
        self.flat_regs[index + self.base] = val


class InterpreterState:
    
    def __init__(self, regs: list[ir.Value], spec: InterpreterStateSpec) -> None:
        super().__init__()
        self.assn_regs = regs.copy()
        self.spec = spec
    
    def ilp_size(self):
        return self.spec.ilp
    
    def __getitem__(self, regset_spec: RegisterSetSpec) -> RegisterSet:
        return RegisterSet(self.assn_regs, regset_spec)
