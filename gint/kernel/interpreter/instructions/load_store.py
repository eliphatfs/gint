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

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        operand = self.op
        load_i = LL.and_(operand, i32(0xf))
        
        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        base_ptr = LL.load(LL.gep(smem_base, [i32(0), i32(0), load_i], inbounds=True))
        base_ptr = LL.bitcast(base_ptr, self.source_dtype.as_pointer())
        offset = LL.lshr(operand, i32(4))
        
        (t_stride, w_stride, o_stride,
         c1_size, c2_size,
         c1_w, c2_w
        ) = [LL.load(LL.gep(smem_base, [i32(0), i32(eid), load_i], inbounds=True)) for eid in range(1, 8)]
        c1_ww, c1_wt, c1_wo = LL.extract_element(c1_w, i32(0)), LL.extract_element(c1_w, i32(1)), LL.extract_element(c1_w, i32(2))
        c2_ww, c2_wt, c2_wo = LL.extract_element(c2_w, i32(0)), LL.extract_element(c2_w, i32(1)), LL.extract_element(c2_w, i32(2))
        c1_ww, c1_wt, c1_wo, c2_ww, c2_wt, c2_wo = [LL.zext(x, i32) for x in [c1_ww, c1_wt, c1_wo, c2_ww, c2_wt, c2_wo]]
        tid = LL.lane_id()
        init_offset = LL.add(LL.mul(offset, o_stride), LL.mul(tid, t_stride))
        c1_cur = LL.add(LL.mul(offset, c1_wo), LL.mul(tid, c1_wt))
        c2_cur = LL.add(LL.mul(offset, c2_wo), LL.mul(tid, c2_wt))
        
        base_ptr = LL.gep(base_ptr, [init_offset], inbounds=True)
        fs = []
        spx = state.peek() if self.mode == 'store' else ([None] * state.reg_width)
        for i in range(0, state.reg_width):
            before_cond = LL.block
            cond_b = LL.and_(
                LL.icmp_signed('<', c1_cur, c1_size),
                LL.icmp_signed('<', c2_cur, c2_size)
            )
            with LL.if_then(cond_b, likely=True):
                # with LL.if_then(LL.icmp_signed('>', LL.logical_program_idx(), i32(49998))):
                #     LL.printf("[b=%d t=%d w=%d] %d %d %d %d %d\n", LL.logical_program_idx(), tid, i32(i), c1_cur, c1_size, c2_cur, c2_size, LL.zext(cond_b, i32))
                if_block = LL.block
                load_val = self.load_store(LL, base_ptr, spx[i])
            if self.mode == 'load':
                # this dominates the final block
                f = LL.phi(f32)
                f.add_incoming(self.oob_value, before_cond)
                f.add_incoming(load_val, if_block)
                fs.append(f)
            if i < state.reg_width - 1:
                c1_cur = LL.add(c1_cur, c1_ww)
                c2_cur = LL.add(c2_cur, c2_ww)
                base_ptr = LL.gep(base_ptr, [w_stride], inbounds=True)
        if self.mode == 'load':
            state.push(fs)
        else:
            state.pop()


class LoadGlobalF32(_LoadStoreGlobalBase):
    source_dtype = f32
    mode = 'load'


class StoreGlobalF32(_LoadStoreGlobalBase):
    source_dtype = f32
    mode = 'store'


class LoadGlobalF16(_LoadStoreGlobalBase):
    source_dtype = f16
    mode = 'load'


class StoreGlobalF16(_LoadStoreGlobalBase):
    source_dtype = f16
    mode = 'store'


class LoadGlobalBF16(_LoadStoreGlobalBase):
    source_dtype = bf16
    mode = 'load'


class StoreGlobalBF16(_LoadStoreGlobalBase):
    source_dtype = bf16
    mode = 'store'


class LoadGlobalU8(_LoadStoreGlobalBase):
    source_dtype = i8
    mode = 'load'


class _LoadStoreGlobalIndirectBase(DefaultControlOperandInstruction):
    source_dtype: ir.Type = void
    oob_value: ir.Constant = f32(0.0)
    mode: str = None

    def load_store(self, LL: PlatformIRBuilder, ptr, store_reg):
        if self.mode == 'load':
            if self.source_dtype == f32:
                return LL.load(ptr)
            raise TypeError("Unsupported type for indirect load", self.source_dtype)
        elif self.mode == 'store':
            if self.source_dtype == f32:
                LL.store(store_reg, ptr)
            else:
                raise TypeError("Unsupported type for indirect store", self.source_dtype)
        else:
            raise ValueError("Unsupported mode", self.mode)

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        operand = self.op
        load_i = LL.and_(operand, i32(0xf))
        
        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        base_ptr = LL.load(LL.gep(smem_base, [i32(0), i32(0), load_i], inbounds=True))
        base_ptr = LL.bitcast(base_ptr, self.source_dtype.as_pointer())
        offset = LL.lshr(operand, i32(4))

        (t_stride, w_stride, o_stride,
         c1_size, c2_size,
         c1_w, c2_w
        ) = [LL.load(LL.gep(smem_base, [i32(0), i32(eid), load_i], inbounds=True)) for eid in range(1, 8)]
        c1_ww, c1_wt, c1_wo = LL.extract_element(c1_w, i32(0)), LL.extract_element(c1_w, i32(1)), LL.extract_element(c1_w, i32(2))
        c2_ww, c2_wt, c2_wo = LL.extract_element(c2_w, i32(0)), LL.extract_element(c2_w, i32(1)), LL.extract_element(c2_w, i32(2))
        c1_ww, c1_wt, c1_wo, c2_ww, c2_wt, c2_wo = [LL.zext(x, i32) for x in [c1_ww, c1_wt, c1_wo, c2_ww, c2_wt, c2_wo]]
        tid = LL.lane_id()
        init_offset = LL.add(LL.mul(offset, o_stride), LL.mul(tid, t_stride))
        c1_cur = LL.add(LL.mul(offset, c1_wo), LL.mul(tid, c1_wt))
        c2_cur = LL.add(LL.mul(offset, c2_wo), LL.mul(tid, c2_wt))
        
        base_ptr = LL.gep(base_ptr, [init_offset], inbounds=True)
        fs = []

        indices_f32 = state.peek(0)
        spx = state.peek(1) if self.mode == 'store' else ([None] * state.reg_width)
        state.pop()
        if self.mode == 'store':
            state.pop()

        indices_i32 = [LL.bitcast(idx, i32) for idx in indices_f32]
        for i in range(0, state.reg_width):
            before_cond = LL.block
            cond_b = LL.and_(
                LL.icmp_signed('<', LL.add(c1_cur, LL.mul(indices_i32[i], c1_wo)), c1_size),
                LL.icmp_signed('<', LL.add(c2_cur, LL.mul(indices_i32[i], c2_wo)), c2_size)
            )
            with LL.if_then(cond_b, likely=True):
                if_block = LL.block
                ptr = LL.gep(base_ptr, [LL.mul(indices_i32[i], o_stride)], inbounds=True)
                load_val = self.load_store(LL, ptr, spx[i])
            if self.mode == 'load':
                # this dominates the final block
                f = LL.phi(f32)
                f.add_incoming(self.oob_value, before_cond)
                f.add_incoming(load_val, if_block)
                fs.append(f)
            if i < state.reg_width - 1:
                c1_cur = LL.add(c1_cur, c1_ww)
                c2_cur = LL.add(c2_cur, c2_ww)
                base_ptr = LL.gep(base_ptr, [w_stride], inbounds=True)
        if self.mode == 'load':
            state.push(fs)



class LoadGlobalF32Indirect(_LoadStoreGlobalIndirectBase):
    source_dtype = f32
    mode = 'load'


class StoreGlobalF32Indirect(_LoadStoreGlobalIndirectBase):
    source_dtype = f32
    mode = 'store'
