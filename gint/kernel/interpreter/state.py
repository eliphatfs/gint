import copy
from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder


class InvalidStateException(Exception):
    pass


class StackMachineState:
    """
    Unified pool of `pool_size` slots. Stack grows up from index 0;
    virtual registers occupy the top `num_regs` slots counting down
    (reg n = pool[pool_size - 1 - n]).  They share the same PHI-node
    budget, so a kernel that uses few registers gets more stack space
    and vice-versa.
    """

    def __init__(self, LL: PlatformIRBuilder, smem_base: ir.Value, pool_size: int, reg_width: int, sp: int, num_regs: int, max_stack: int):
        self.pc = LL.phi(i32.as_pointer())
        self.opcode = LL.phi(i32)
        self.pool = [[LL.phi(f32) for _ in range(reg_width)] for _ in range(pool_size)]
        self.sp = sp
        self.pool_size = pool_size
        self.num_regs = num_regs
        self.max_stack = max_stack
        self.reg_width = reg_width
        self.smem_base = smem_base

    # ------------------------------------------------------------------
    # Properties that expose the two logical regions of the pool
    # ------------------------------------------------------------------

    @property
    def stack(self):
        """Stack slots pool[0..max_stack-1]."""
        return self.pool[:self.max_stack]

    @property
    def regs(self):
        """Register file: reg n → pool[pool_size - 1 - n]."""
        return [self.pool[self.pool_size - 1 - n] for n in range(self.num_regs)]

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def clone(self):
        state = copy.copy(self)
        state.pool = [copy.copy(self.pool[i]) for i in range(self.pool_size)]
        return state

    def push(self, values: list):
        assert len(values) == self.reg_width
        if self.sp >= self.max_stack:
            raise InvalidStateException('push')
        self.pool[self.sp] = values
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
        return self.pool[self.sp - 1 - topx]

    def flat_regs(self):
        flat = [self.pc, self.opcode]
        for sub in self.pool:
            flat.extend(sub)
        return flat
