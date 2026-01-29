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
        idx_t = LL.lshr(operand, i32(4))
        
        ilp_s, ilp_sz, t_s, t_sz, i_tofs_s = [LL.load(LL.gep(smem_base, [i32(0), i32(eid), load_i], inbounds=True)) for eid in range(1, 6)]
        t_base = LL.add(idx_t, LL.lane_id())
        
        base_ptr = LL.gep(base_ptr, [LL.mul(t_s, t_base)], inbounds=True)
        fs = []
        spx = state.peek() if self.mode == 'store' else ([None] * state.reg_width)
        for i in range(0, state.reg_width):
            before_cond = LL.block
            cond_b = LL.icmp_unsigned('<', t_base, t_sz)
            if i > 0:
                ilp_t = LL.icmp_unsigned('<', i32(i), ilp_sz)
                cond_b = LL.and_(ilp_t, cond_b)
            with LL.if_then(cond_b, likely=True):
                if_block = LL.block
                load_val = self.load_store(LL, base_ptr, spx[i])
            if self.mode == 'load':
                # this dominates the final block
                f = LL.phi(f32)
                f.add_incoming(self.oob_value, before_cond)
                f.add_incoming(load_val, if_block)
                fs.append(f)
            if i < state.reg_width - 1:
                t_base = LL.add(t_base, i_tofs_s)
                base_ptr = LL.gep(base_ptr, [ilp_s], inbounds=True)
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
        
        if isinstance(self.source_dtype, ir.IntType):
            elm_sz_val = i32(self.source_dtype.width // 8)
        elif isinstance(self.source_dtype, ir.HalfType):
            elm_sz_val = i32(2)
        elif isinstance(self.source_dtype, ir.FloatType):
            elm_sz_val = i32(4)
        elif isinstance(self.source_dtype, ir.DoubleType):
            elm_sz_val = i32(8)
        else:
            # Fallback for BFloat16 or others if they have width or similar
            elm_sz_val = i32(getattr(self.source_dtype, 'width', 32) // 8)
        
        # Get index from stack (top of stack)
        indices_f32 = state.peek(0)
        
        # Get values if store (under the index)
        values_f32 = state.peek(1) if self.mode == 'store' else None

        # Pop from stack
        state.pop()
        if self.mode == 'store':
            state.pop()

        indices_i32 = [LL.bitcast(idx, i32) for idx in indices_f32]
        
        results = []
        for i in range(state.reg_width):
            offset = LL.mul(LL.zext(indices_i32[i], i64), LL.zext(elm_sz_val, i64))
            ptr = LL.gep(base_ptr, [offset], inbounds=True)
            ptr = LL.bitcast(ptr, self.source_dtype.as_pointer())
            
            val = self.load_store(LL, ptr, values_f32[i] if values_f32 else None)
            if self.mode == 'load':
                results.append(val)
        
        if self.mode == 'load':
            state.push(results)


class LoadGlobalF32Indirect(_LoadStoreGlobalIndirectBase):
    source_dtype = f32
    mode = 'load'


class StoreGlobalF32Indirect(_LoadStoreGlobalIndirectBase):
    source_dtype = f32
    mode = 'store'
