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

    Register-spill window (`spill_window` > 0): only the top ``spill_window``
    stack slots ("the window") plus all virtual registers are kept in
    VGPRs and threaded through the per-stack-depth dispatch-block PHI web.
    Stack slots below the window are *spilled to per-warp shared memory*
    (smem) whenever ``sp`` rises above ``slot + spill_window``, and reloaded
    when ``sp`` drops back.  This breaks the cross-dispatch-block PHI web
    for the preserved-but-untouched low slots, which otherwise forces LLVM's
    coalescer to keep each such slot in a distinct VGPR across all mutually
    exclusive high-sp dispatch regions (VGPR blow-up + scratch spills).

    The window must be larger than the deepest ``peek()`` depth used by any
    instruction (the interpreter's deepest read is ``peek(2)`` via DupX2), so
    ``spill_window >= 3`` is the correctness floor; we default to 4 to leave
    one slot of headroom.  ``spill_window == 0`` disables the scheme entirely
    (all slots in VGPRs, classic behaviour) — used by the NVPTX backend and
    as an off-switch.
    """

    def __init__(self, LL: PlatformIRBuilder, smem_base: ir.Value, pool_size: int, reg_width: int, sp: int, num_regs: int, max_stack: int, spill_window: int = 0):
        self.pc = LL.phi(i32.as_pointer())
        self.opcode = LL.phi(i32)
        # spill window (0 disables; must exceed the interpreter's max peek
        # depth of 2 when enabled, i.e. >= 3).
        self.spill_window = spill_window
        self.smem_base = smem_base
        self.sp = sp
        self.pool_size = pool_size
        self.num_regs = num_regs
        self.max_stack = max_stack
        self.reg_width = reg_width
        # Keep the IR builder so push/pop can emit smem spill/reload at the
        # current insertion point (an instruction's case block).  It is the
        # same builder instance the instructions use; its position is advanced
        # by main.py / the instruction emit before push/pop is called.
        self._LL = LL
        # pool[k] is either a list[reg_width] of register SSA values (PHI or
        # computed) or None when slot k currently lives in smem (spilled).
        self.pool = []
        for k in range(pool_size):
            if self._is_active(k, sp):
                self.pool.append([LL.phi(f32) for _ in range(reg_width)])
            else:
                self.pool.append(None)

    # ------------------------------------------------------------------
    # Active-slot set: window (top `spill_window` stack slots) + registers
    # ------------------------------------------------------------------

    def _is_active(self, k: int, sp: int) -> bool:
        """True if slot k is held in a VGPR (PHI-threaded) at stack depth sp.

        The top ``num_regs`` pool slots are the register file (reg n ->
        pool[pool_size-1-n]); fload_reg/fstore_reg touch them at ANY sp, so
        register slots are always active (never spilled).  Pure stack slots
        (k < pool_size - num_regs) are active only while inside the top
        ``spill_window`` window; below it they live in smem.  spill_window==0
        disables the scheme -> every stack slot (k < sp) is active (classic
        all-in-VGPR behaviour, no smem spill).

        Note the unified pool: register slots overlap the high stack slots,
        so when sp is high the window lands entirely on register slots (which
        are already active) and only the pure-stack slots below are spilled."""
        if k >= self.pool_size - self.num_regs:
            return True
        if self.spill_window <= 0:
            return k < sp   # classic: all live stack slots in VGPRs
        return k < sp and k >= sp - self.spill_window

    def active_slots(self, sp: int) -> list:
        """Deterministic ascending list of pool indices that are in VGPRs at
        stack depth ``sp``.  Used by ``flat_regs`` and by the dispatch-block
        PHI creation so the two stay in lock-step."""
        return [k for k in range(self.pool_size) if self._is_active(k, sp)]

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

    # ---- smem spill/reload helpers (no-op when spill_window == 0) ----

    def _spill_slot_stride_bytes(self) -> int:
        # byte stride per lane for one spilled slot = reg_width * sizeof(f32).
        return self.reg_width * 4

    def _spill_slots_this_variant(self) -> int:
        # max number of PURE-STACK slots a single lane can have spilled
        # simultaneously for this variant.  A pure-stack slot k (k <
        # pool_size-num_regs) is spilled when it sits below the window
        # (k < sp - W).  At peak sp=max_stack that is:
        #   k < min(pool_size-num_regs, max_stack - W)
        return max(0, min(self.pool_size - self.num_regs,
                           self.max_stack - self.spill_window))

    def _spill_addr(self, LL: PlatformIRBuilder, slot: int):
        """smem address of spilled slot `slot` for the calling lane.

        Layout (per warp, laid out after the existing tensor-info region in
        SMEM_PER_WARP): a 2D array [wave_size][spill_slots] of f32[reg_width].
        Lane l, slot k -> base + (l*spill_slots + k) * (reg_width * 4) bytes.
        Each lane writes its own disjoint region -> no cross-lane aliasing."""
        if self.spill_window <= 0:
            return None
        # smem_base is i8 addrspace(3)* already offset to this warp's region.
        # Step past the tensor-info region (byte offset) to the spill area.
        from .main import SMEM_TENSOR_INFO_BYTES  # tensor-info region size
        spill_slots = self._spill_slots_this_variant()
        per_lane_bytes = spill_slots * self._spill_slot_stride_bytes()
        lane = LL.lane_id()
        byte_offset = LL.add(LL.mul(lane, i32(per_lane_bytes)),
                             i32(SMEM_TENSOR_INFO_BYTES + slot * self._spill_slot_stride_bytes()))
        ptr_i8 = LL.gep(self.smem_base, [byte_offset], inbounds=True)
        return LL.bitcast(ptr_i8, f32.as_pointer(LL.smem_addrspace()))

    def _spill(self, LL: PlatformIRBuilder, slot: int):
        """Store pool[slot] (reg_width f32) into smem[slot] and drop it from
        VGPRs (pool[slot] = None)."""
        vals = self.pool[slot]
        assert vals is not None, "spilling an already-spilled slot"
        ptr = self._spill_addr(LL, slot)
        for w in range(self.reg_width):
            LL.store(vals[w], LL.gep(ptr, [i32(w)], inbounds=True))
        self.pool[slot] = None

    def _reload(self, LL: PlatformIRBuilder, slot: int):
        """Load reg_width f32 from smem[slot] back into pool[slot] (VGPR)."""
        ptr = self._spill_addr(LL, slot)
        self.pool[slot] = [LL.load(LL.gep(ptr, [i32(w)], inbounds=True))
                           for w in range(self.reg_width)]

    def push(self, values: list):
        assert len(values) == self.reg_width
        if self.sp >= self.max_stack:
            raise InvalidStateException('push')
        # The slot that leaves the window (old window bottom) spills to smem,
        # but only if it is a PURE-STACK slot (register slots stay active).
        # Before push: sp=S, window = pool[S-1 .. S-W].  After: sp=S+1, window
        # = pool[S .. S-W+1]; pool[S-W] drops below -> spill it.
        leaving = self.sp - self.spill_window
        if self.spill_window > 0 and 0 <= leaving < self.pool_size - self.num_regs:
            assert self.pool[leaving] is not None
            self._spill(self._LL, leaving)
        self.pool[self.sp] = values
        self.sp += 1
        return self

    def pop(self):
        if self.sp <= 0:
            raise InvalidStateException('pop')
        self.sp -= 1
        # The slot that enters the window (new window bottom) reloads from smem,
        # but only if it is a PURE-STACK slot (register slots stay active).
        # After pop: sp=S-1, window = pool[S-2 .. S-1-W]; pool[S-1-W] enters.
        entering = self.sp - self.spill_window
        if self.spill_window > 0 and 0 <= entering < self.pool_size - self.num_regs:
            assert self.pool[entering] is None, "reloading a non-spilled slot"
            self._reload(self._LL, entering)
        return self

    def peek(self, topx: int = 0):
        assert topx >= 0
        if self.spill_window > 0:
            assert topx < self.spill_window, (
                "peek depth %d >= spill_window %d (slot would be in smem)" % (topx, self.spill_window))
        if self.sp - 1 - topx < 0:
            raise InvalidStateException('peek', topx)
        return self.pool[self.sp - 1 - topx]

    def flat_regs(self):
        flat = [self.pc, self.opcode]
        for k in self.active_slots(self.sp):
            flat.extend(self.pool[k])
        return flat
