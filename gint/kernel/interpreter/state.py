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


class AllocaPool:
    """Indexed view into an alloca'd pool array. Supports __getitem__ and
    __setitem__ with load/store semantics so it behaves like the PHI-based
    pool list for register instructions that access pool slots directly."""

    def __init__(self, LL, base_ptr, pool_size, reg_width):
        self.LL = LL
        self.base = base_ptr
        self.pool_size = pool_size
        self.reg_width = reg_width

    def __getitem__(self, idx):
        base = idx * self.reg_width
        return [self.LL.load(self.LL.gep(self.base, [i32(base + j)], inbounds=True))
                for j in range(self.reg_width)]

    def __setitem__(self, idx, values):
        base = idx * self.reg_width
        for j, v in enumerate(values):
            ptr = self.LL.gep(self.base, [i32(base + j)], inbounds=True)
            self.LL.store(v, ptr)


class AllocaStackMachineState:
    """
    Variant of StackMachineState where sp is an alloca'd runtime value
    and pool values live in an alloca'd local-memory array instead of
    PHI nodes.  This removes stack-depth specialization: there is a
    single dispatch block for all stack depths.

    Stack grows up from index 0; virtual registers occupy the top
    `num_regs` slots counting down (reg n = pool[pool_size - 1 - n]).
    """

    def __init__(self, LL: PlatformIRBuilder, smem_base: ir.Value, pool_size: int, reg_width: int, sp_init: int, num_regs: int, max_stack: int, sp_alloca=None, pool_alloca=None):
        self.LL = LL
        self.pc = LL.phi(i32.as_pointer())
        self.opcode = LL.phi(i32)
        self._pool_alloca = pool_alloca if pool_alloca is not None else LL.alloca(f32, size=pool_size * reg_width, name="pool")
        self.pool = AllocaPool(LL, self._pool_alloca, pool_size, reg_width)
        self.sp = sp_alloca if sp_alloca is not None else LL.alloca(i32, name="sp")
        if sp_alloca is None:
            LL.store(i32(sp_init), self.sp)
        self.pool_size = pool_size
        self.num_regs = num_regs
        self.max_stack = max_stack
        self.reg_width = reg_width
        self.smem_base = smem_base

    @property
    def regs(self):
        """Register file: reg n → pool[pool_size - 1 - n] (loaded from alloca)."""
        result = []
        for n in range(self.num_regs):
            base = (self.pool_size - 1 - n) * self.reg_width
            row = []
            for j in range(self.reg_width):
                ptr = self.LL.gep(self._pool_alloca, [i32(base + j)], inbounds=True)
                row.append(self.LL.load(ptr))
            result.append(row)
        return result

    def clone(self):
        return copy.copy(self)

    def push(self, values: list):
        assert len(values) == self.reg_width
        sp_val = self.LL.load(self.sp)
        base = self.LL.mul(sp_val, i32(self.reg_width))
        for j, v in enumerate(values):
            idx = self.LL.add(base, i32(j))
            ptr = self.LL.gep(self._pool_alloca, [idx], inbounds=True)
            self.LL.store(v, ptr)
        self.LL.store(self.LL.add(sp_val, i32(1)), self.sp)
        return self

    def pop(self):
        sp_val = self.LL.load(self.sp)
        self.LL.store(self.LL.sub(sp_val, i32(1)), self.sp)
        return self

    def peek(self, topx: int = 0):
        assert topx >= 0
        sp_val = self.LL.load(self.sp)
        idx_base = self.LL.sub(sp_val, i32(1 + topx))
        base = self.LL.mul(idx_base, i32(self.reg_width))
        result = []
        for j in range(self.reg_width):
            idx = self.LL.add(base, i32(j))
            ptr = self.LL.gep(self._pool_alloca, [idx], inbounds=True)
            result.append(self.LL.load(ptr))
        return result

    def flat_regs(self):
        return [self.pc, self.opcode]
