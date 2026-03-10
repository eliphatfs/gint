from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction
from ...platforms.common import *


def _make_load_reg(n: int):
    class FLoadRegN(DefaultControlInstruction):
        __doc__ = f"Push a copy of virtual register {n} onto the stack."
        _reg_n = n

        def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
            k = state.pool_size - 1 - self._reg_n
            state.push(list(state.pool[k]))

    FLoadRegN.__name__ = f"FLoadReg{n}"
    FLoadRegN.__qualname__ = f"FLoadReg{n}"
    return FLoadRegN


def _make_store_reg(n: int):
    class FStoreRegN(DefaultControlInstruction):
        __doc__ = f"Pop the top of the stack and write to virtual register {n}."
        _reg_n = n

        def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
            v = state.peek()
            state.pop()
            k = state.pool_size - 1 - self._reg_n
            state.pool[k] = list(v)

    FStoreRegN.__name__ = f"FStoreReg{n}"
    FStoreRegN.__qualname__ = f"FStoreReg{n}"
    return FStoreRegN


# Generate specialized classes for registers 0-7
FLoadReg0  = _make_load_reg(0)
FLoadReg1  = _make_load_reg(1)
FLoadReg2  = _make_load_reg(2)
FLoadReg3  = _make_load_reg(3)
FLoadReg4  = _make_load_reg(4)
FLoadReg5  = _make_load_reg(5)
FLoadReg6  = _make_load_reg(6)
FLoadReg7  = _make_load_reg(7)

FStoreReg0 = _make_store_reg(0)
FStoreReg1 = _make_store_reg(1)
FStoreReg2 = _make_store_reg(2)
FStoreReg3 = _make_store_reg(3)
FStoreReg4 = _make_store_reg(4)
FStoreReg5 = _make_store_reg(5)
FStoreReg6 = _make_store_reg(6)
FStoreReg7 = _make_store_reg(7)

LOAD_REGS  = [FLoadReg0,  FLoadReg1,  FLoadReg2,  FLoadReg3,
              FLoadReg4,  FLoadReg5,  FLoadReg6,  FLoadReg7]
STORE_REGS = [FStoreReg0, FStoreReg1, FStoreReg2, FStoreReg3,
              FStoreReg4, FStoreReg5, FStoreReg6, FStoreReg7]


class FRcp(DefaultControlInstruction):
    """Pop x, push 1/x (reciprocal)."""

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        reg = state.peek()
        state.pop().push([LL.fdiv(f32(1.0), x) for x in reg])
