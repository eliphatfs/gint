import copy
from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder


class InvalidStateException(Exception):
    pass


class StackMachineState:
    
    def __init__(self, LL: PlatformIRBuilder, smem_base: ir.Value, max_stack: int, reg_width: int, sp: int):
        self.pc = LL.phi(i32.as_pointer())
        self.opcode = LL.phi(i32)
        self.stack = [[LL.phi(f32) for _ in range(reg_width)] for _ in range(max_stack)]
        self.sp = sp
        self.max_stack = max_stack
        self.reg_width = reg_width
        self.smem_base = smem_base

    def clone(self):
        state = copy.copy(self)
        state.stack = [copy.copy(self.stack[i]) for i in range(self.max_stack)]
        return state
    
    def push(self, values: list):
        assert len(values) == self.reg_width
        if self.sp >= len(self.stack):
            raise InvalidStateException('push')
        self.stack[self.sp] = values
        self.sp += 1
        return self

    def pop(self):
        if self.sp <= 0:
            raise InvalidStateException('pop')
        self.sp -= 1
        return self

    def peek(self, topx: int = 0):
        assert topx >= 0
        if self.sp - 1 - topx < 0:
            raise InvalidStateException('peek', topx)
        return self.stack[self.sp - 1 - topx]

    def flat_regs(self):
        flat = [self.pc, self.opcode]
        for sub in self.stack:
            flat.extend(sub)
        return flat
