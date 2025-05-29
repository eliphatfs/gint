from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


BlockTensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, 8),  # ptr with block offset
    ir.ArrayType(i32, 8),  # ilp stride
    ir.ArrayType(i32, 8),  # ilp size
    ir.ArrayType(i32, 8),  # thread stride
    ir.ArrayType(i32, 8),  # thread size
    ir.ArrayType(i32, 8),  # ilp contribution stride to thread offset
])


class LoadTensorInfos(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        lane_id = LL.lane_id()
        rt_ofs = i32(0)
        with LL.if_then(LL.icmp_signed('<', lane_id, LL.arg(2))):
            base_ptr, b_strides, b_sizes, b_tofs_stride, t_stride, t_size, elm_sz = [
                LL.load(LL.gep(LL.arg(1), [i32(0), i32(eid), lane_id], inbounds=True))
                for eid in range(7)
            ]
            bidx = LL.logical_program_idx()
            rbidx = bidx
            for i in range(4):
                bs = LL.extract_element(b_strides, i32(i))
                sz = LL.extract_element(b_sizes, i32(i))
                btofss = LL.extract_element(b_tofs_stride, i32(i))
                if i == 0:
                    # ilp block size
                    ilp_s = bs
                    ilp_sz = sz
                    ilp_ofs_stride = btofss
                    sz = LL.udiv(LL.add(sz, i32(ispec.ilp - 1)), i32(ispec.ilp))  # ceil div
                idx = LL.urem(rbidx, sz)
                rbidx = LL.udiv(rbidx, sz)
                if i == 0:
                    idx = LL.mul(idx, i32(ispec.ilp))
                    ilp_sz = LL.sub(ilp_sz, idx)
                base_ptr = LL.gep(base_ptr, [LL.mul(LL.mul(LL.zext(bs, i64), LL.zext(idx, i64)), LL.zext(elm_sz, i64))], inbounds=True)
                rt_ofs = LL.add(rt_ofs, LL.mul(idx, btofss))
            t_size = LL.sub(t_size, rt_ofs)
            smem_base = state.smem_base
            smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
            for eid, val in enumerate([base_ptr, ilp_s, ilp_sz, t_stride, t_size, ilp_ofs_stride]):
                LL.store(val, LL.gep(smem_base, [i32(0), i32(eid), lane_id], inbounds=True))
        LL.warp_sync()


class _LoadStoreGlobalBase(Instruction):
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

    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        operand = state.operand
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
        for i in range(0, ispec.ilp):
            before_cond = LL.block
            cond_b = LL.icmp_unsigned('<', t_base, t_sz)
            if i > 0:
                ilp_t = LL.icmp_unsigned('<', i32(i), ilp_sz)
                cond_b = LL.and_(ilp_t, cond_b)
            with LL.if_then(cond_b, likely=True):
                if_block = LL.block
                load_val = self.load_store(LL, base_ptr, state[ispec.rf0][i])
            if self.mode == 'load':
                # this dominates the final block
                f = LL.phi(f32)
                f.add_incoming(self.oob_value, before_cond)
                f.add_incoming(load_val, if_block)
                fs.append(f)
            if i < ispec.ilp - 1:
                t_base = LL.add(t_base, i_tofs_s)
                base_ptr = LL.gep(base_ptr, [ilp_s], inbounds=True)
        if self.mode == 'load':
            state[ispec.rf1] = fs


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
