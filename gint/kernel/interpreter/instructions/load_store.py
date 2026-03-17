from types import NotImplementedType
from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlOperandInstruction
from .load_tensor_infos import BlockTensorInfo
from ...platforms.common import *


class _LoadStoreGlobalBase(DefaultControlOperandInstruction):
    source_dtype: ir.Type = void
    mode: str = None
    oob_value: ir.Constant = f32(0.0)
    
    def load_store(self, LL: PlatformIRBuilder, ptr, store_reg):
        if self.mode == 'load':
            if self.source_dtype == f32:
                return LL.load(ptr)
            if self.source_dtype == f16:
                return LL.fpext(LL.load(ptr), f32)
            if self.source_dtype == bf16:
                return LL.fpext(LL.load(ptr), f32)
            if self.source_dtype == i8:
                return LL.uitofp(LL.load(ptr), f32)
            raise TypeError("Unsupported type (need f32, f16, i8 or bf16)", self.source_dtype)
        elif self.mode == 'store':
            if self.source_dtype == f32:
                return LL.store(store_reg, ptr)
            if self.source_dtype == f16:
                return LL.store(LL.fptrunc(store_reg, f16), ptr)
            if self.source_dtype == bf16:
                return LL.store(LL.fptrunc(store_reg, bf16), ptr)
            raise TypeError("Unsupported type (need f32, f16 or bf16)", self.source_dtype)
        else:
            raise ValueError("Unsupported mode (need load or store)", self.mode)

    def init_block(self, LL: PlatformIRBuilder, state: StackMachineState, load_i: ir.Value, smem_base: ir.Value):
        raise NotImplementedError

    def init_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, offset: ir.Value):
        raise NotImplementedError

    def advance_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, w: int, wstate: ir.Value):
        raise NotImplementedError

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        operand = self.op
        load_i = LL.and_(operand, i32(0xf))
        
        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        base_ptr = LL.load(LL.gep(smem_base, [i32(0), i32(0), load_i], inbounds=True))
        base_ptr = LL.bitcast(base_ptr, self.source_dtype.as_pointer())
        offset = LL.lshr(operand, i32(4))

        block = self.init_block(LL, state, load_i, smem_base)
        base_ptr, cond_b, wstate = self.init_ptr_cond_state(LL, state, base_ptr, block, offset)
        
        spx = state.peek() if self.mode == 'store' else ([None] * state.reg_width)
        fs = []
        for w in range(state.reg_width):
            before_cond = LL.block
            with LL.if_then(cond_b, likely=True):
                if_block = LL.block
                load_val = self.load_store(LL, base_ptr, spx[w])
            if self.mode == 'load':
                # this dominates the final block
                f = LL.phi(f32)
                f.add_incoming(self.oob_value, before_cond)
                f.add_incoming(load_val, if_block)
                fs.append(f)
            if w < state.reg_width - 1:
                base_ptr, cond_b, wstate = self.advance_ptr_cond_state(LL, state, base_ptr, block, w + 1, wstate)
        if self.mode == 'load':
            state.push(fs)
        else:
            state.pop()


class _LoadStoreGlobalBase1D(_LoadStoreGlobalBase):

    def init_block(self, LL: PlatformIRBuilder, state: StackMachineState, load_i: ir.Value, smem_base: ir.Value):
        (block_1,) = [
            LL.load(LL.gep(smem_base, [i32(0), i32(eid), load_i], inbounds=True))
            for eid in [1]
        ]
        block_shape, block_stride = LL.extract_element(block_1, i32(0)), LL.extract_element(block_1, i32(1))
        return block_shape, block_stride

    def init_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, offset: ir.Value):
        block_shape, block_stride = block
        t = LL.lane_id()
        init_offset_total = LL.add(t, offset)
        base_ptr = LL.gep(base_ptr, [LL.mul(init_offset_total, block_stride)], inbounds=True)
        return base_ptr, LL.icmp_signed('<', init_offset_total, block_shape), (init_offset_total, LL.mul(block_stride, LL.warp_size()))

    def advance_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, w: int, wstate: ir.Value):
        block_shape, _ = block
        offset, w_stride = wstate
        base_ptr = LL.gep(base_ptr, [w_stride], inbounds=True)
        offset = LL.add(offset, LL.warp_size())
        return base_ptr, LL.icmp_signed('<', offset, block_shape), (offset, w_stride)


class _LoadStoreGlobalBase2DT(_LoadStoreGlobalBase):

    def init_block(self, LL: PlatformIRBuilder, state: StackMachineState, load_i: ir.Value, smem_base: ir.Value):
        (block_1, block_2, adv_offset) = [
            LL.load(LL.gep(smem_base, [i32(0), i32(eid), load_i], inbounds=True))
            for eid in [1, 2, 3]
        ]
        block_shape_1, block_stride_1 = LL.extract_element(block_1, i32(0)), LL.extract_element(block_1, i32(1))
        block_shape_2, block_stride_2 = LL.extract_element(block_2, i32(0)), LL.extract_element(block_2, i32(1))
        return block_shape_1, block_stride_1, block_shape_2, block_stride_2, adv_offset

    def offsets_t_w(self, LL: PlatformIRBuilder, offset: ir.Value, adv_offset: ir.Value):
        t = LL.lane_id()
        offset_t = LL.add(t, offset)
        init_offset_w = adv_offset
        return offset_t, init_offset_w

    def init_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, offset: ir.Value):
        block_shape_1, block_stride_1, block_shape_2, block_stride_2, adv_offset = block
        offset_t, init_offset_w = self.offsets_t_w(LL, offset, adv_offset)
        base_ptr = LL.gep(
            base_ptr,
            [LL.add(
                LL.mul(offset_t, block_stride_1),
                LL.mul(init_offset_w, block_stride_2)
            )], inbounds=True
        )
        t_cond = LL.icmp_signed('<', offset_t, block_shape_1)
        return (
            base_ptr,
            LL.and_(
                t_cond,
                LL.icmp_signed('<', init_offset_w, block_shape_2)
            ),
            (t_cond, init_offset_w)
        )

    def advance_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, w: int, wstate: ir.Value):
        block_shape_1, block_stride_1, block_shape_2, block_stride_2, adv_offset = block
        t_cond, offset_w = wstate
        base_ptr = LL.gep(base_ptr, [block_stride_2], inbounds=True)
        offset_w = LL.add(offset_w, i32(1))
        return (
            base_ptr,
            LL.and_(t_cond, LL.icmp_signed('<', offset_w, block_shape_2)),
            (t_cond, offset_w)
        )


class _LoadStoreGlobalBase2DW(_LoadStoreGlobalBase2DT):
    def offsets_t_w(self, LL: PlatformIRBuilder, offset: ir.Value, adv_offset: ir.Value):
        t = LL.lane_id()
        offset_t = LL.add(t, adv_offset)
        init_offset_w = offset
        return offset_t, init_offset_w


class LoadGlobal2DTF32(_LoadStoreGlobalBase2DT):
    source_dtype = f32
    mode = 'load'


class StoreGlobal2DTF32(_LoadStoreGlobalBase2DT):
    source_dtype = f32
    mode = 'store'


class LoadGlobal2DTF16(_LoadStoreGlobalBase2DT):
    source_dtype = f16
    mode = 'load'


class StoreGlobal2DTF16(_LoadStoreGlobalBase2DT):
    source_dtype = f16
    mode = 'store'


class LoadGlobal2DTBF16(_LoadStoreGlobalBase2DT):
    source_dtype = bf16
    mode = 'load'


class StoreGlobal2DTBF16(_LoadStoreGlobalBase2DT):
    source_dtype = bf16
    mode = 'store'


class LoadGlobal2DTU8(_LoadStoreGlobalBase2DT):
    source_dtype = i8
    mode = 'load'


class LoadGlobal2DWF32(_LoadStoreGlobalBase2DW):
    source_dtype = f32
    mode = 'load'


class StoreGlobal2DWF32(_LoadStoreGlobalBase2DW):
    source_dtype = f32
    mode = 'store'


class LoadGlobal2DWF16(_LoadStoreGlobalBase2DW):
    source_dtype = f16
    mode = 'load'


class StoreGlobal2DWF16(_LoadStoreGlobalBase2DW):
    source_dtype = f16
    mode = 'store'


class LoadGlobal2DWBF16(_LoadStoreGlobalBase2DW):
    source_dtype = bf16
    mode = 'load'


class StoreGlobal2DWBF16(_LoadStoreGlobalBase2DW):
    source_dtype = bf16
    mode = 'store'


class LoadGlobal2DWU8(_LoadStoreGlobalBase2DW):
    source_dtype = i8
    mode = 'load'


class LoadGlobal1DF32(_LoadStoreGlobalBase1D):
    source_dtype = f32
    mode = 'load'


class StoreGlobal1DF32(_LoadStoreGlobalBase1D):
    source_dtype = f32
    mode = 'store'


class LoadGlobal1DF16(_LoadStoreGlobalBase1D):
    source_dtype = f16
    mode = 'load'


class StoreGlobal1DF16(_LoadStoreGlobalBase1D):
    source_dtype = f16
    mode = 'store'


class LoadGlobal1DBF16(_LoadStoreGlobalBase1D):
    source_dtype = bf16
    mode = 'load'


class StoreGlobal1DBF16(_LoadStoreGlobalBase1D):
    source_dtype = bf16
    mode = 'store'


class LoadGlobal1DU8(_LoadStoreGlobalBase1D):
    source_dtype = i8
    mode = 'load'


class _LoadStoreGlobalIndirect1D(_LoadStoreGlobalBase1D):

    def init_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, offset: ir.Value):
        indices_f32 = state.peek()
        state.pop()
        return self.advance_ptr_cond_state(LL, state, base_ptr, block, 0, (base_ptr, indices_f32))

    def advance_ptr_cond_state(self, LL: PlatformIRBuilder, state: StackMachineState, base_ptr: ir.Value, block: ir.Value, w: int, wstate: ir.Value):
        block_shape, block_stride = block
        base_ptr, indices_f32 = wstate
        offset = LL.bitcast(indices_f32[w], i32)
        ptr = LL.gep(base_ptr, [LL.mul(offset, block_stride)], inbounds=True)
        return ptr, LL.icmp_signed('<', offset, block_shape), (base_ptr, indices_f32)


class LoadGlobal1DF32Indirect(_LoadStoreGlobalIndirect1D):
    source_dtype = f32
    mode = 'load'


class StoreGlobal1DF32Indirect(_LoadStoreGlobalIndirect1D):
    source_dtype = f32
    mode = 'store'


class AdvanceBlock2D(DefaultControlOperandInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        operand = self.op
        load_i = LL.and_(operand, i32(0xf))
        offset = LL.ashr(operand, i32(4))
        
        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        
        with LL.if_then(LL.icmp_signed('==', LL.lane_id(), i32(0))):
            adv_offset_ptr = LL.gep(smem_base, [i32(0), i32(3), load_i], inbounds=True)
            old_val = LL.load(adv_offset_ptr)
            LL.store(LL.add(old_val, offset), adv_offset_ptr)
        LL.warp_sync()


class AdvanceBase(DefaultControlOperandInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        operand = self.op
        load_i = LL.and_(operand, i32(0xf))
        offset = LL.ashr(operand, i32(4))
        
        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        
        with LL.if_then(LL.icmp_signed('==', LL.lane_id(), i32(0))):
            base_ptr_ptr = LL.gep(smem_base, [i32(0), i32(0), load_i], inbounds=True)
            old_val = LL.load(base_ptr_ptr)
            # Offset is in bytes
            LL.store(LL.gep(old_val, [offset], inbounds=True), base_ptr_ptr)
        LL.warp_sync()
