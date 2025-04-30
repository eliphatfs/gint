from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


class LoadTensorInfos(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        lane_id = LL.lane_id()
        before_block = LL.block
        with LL.if_then(LL.icmp_signed('<', lane_id, LL.arg(2))):
            if_block = LL.block
            b_strides, b_sizes, t_stride, t_size, elm_sz, resv_0, resv_1, resv_2, base_ptr = [
                LL.load(LL.gep(LL.arg(1), [lane_id, i32(eid)], inbounds=True))
                for eid in range(9)
            ]
            bidx = LL.block_idx_x()
            rbidx = bidx
            for i in range(4):
                bs = LL.extract_element(b_strides, i32(i))
                sz = LL.extract_element(b_sizes, i32(i))
                if i == 0:
                    # ilp block size
                    ilp_s = bs
                    ilp_sz = sz
                    sz = LL.udiv(LL.add(sz, i32(ispec.ilp - 1)), i32(ispec.ilp))  # ceil div
                idx = LL.urem(rbidx, sz)
                rbidx = LL.udiv(rbidx, sz)
                if i == 0:
                    idx = LL.mul(idx, i32(ispec.ilp))
                    ilp_sz = LL.sub(ilp_sz, idx)
                base_ptr = LL.gep(base_ptr, [LL.mul(LL.mul(LL.zext(bs, i64), LL.zext(idx, i64)), LL.zext(elm_sz, i64))], inbounds=True)
        rts = LL.phi(i32)
        rts.add_incoming(t_stride, if_block)
        rts.add_incoming(i32(0), before_block)
        rtsz = LL.phi(i32)
        rtsz.add_incoming(t_size, if_block)
        rtsz.add_incoming(i32(0), before_block)
        ilps = LL.phi(i32)
        ilps.add_incoming(ilp_s, if_block)
        ilps.add_incoming(i32(0), before_block)
        ilpsz = LL.phi(i32)
        ilpsz.add_incoming(ilp_sz, if_block)
        ilpsz.add_incoming(i32(0), before_block)
        ptr = LL.phi(p_i8g)
        ptr.add_incoming(base_ptr, if_block)
        ptr.add_incoming(p_i8g(None), before_block)
        state[ispec.rof][0] = ptr
        state[ispec.rss] = [ilps, ilpsz, rts, rtsz]


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
        idx_t = LL.lshr(operand, i32(4))
        base_ptr = LL.warp_broadcast_lane_b64_as_2xi32(state[ispec.rof][0], load_i)
        base_ptr = LL.bitcast(base_ptr, self.source_dtype.as_pointer(1))
        
        t_base = LL.add(idx_t, LL.thread_idx_x())
        
        ilps, ilpsz, rts, rtsz = state[ispec.rss]
        t_sz = LL.warp_broadcast_lane(rtsz, load_i)
        t_s = LL.warp_broadcast_lane(rts, load_i)
        
        ilp_s = LL.warp_broadcast_lane(ilps, load_i)
        ilp_sz = LL.warp_broadcast_lane(ilpsz, load_i)
        
        beforeall_block = LL.block
        with LL.if_then(LL.icmp_unsigned('<', t_base, t_sz), likely=True):
            base_ptr = LL.gep(base_ptr, [LL.mul(t_s, t_base)])
            f0 = self.load_store(LL, base_ptr, state[ispec.rf0][0])
            fs = [f0]
            for i in range(1, ispec.ilp):
                before_cond = LL.block
                base_ptr = LL.gep(base_ptr, [ilp_s])
                with LL.if_then(LL.icmp_signed('<', i32(i), ilp_sz), likely=True):
                    if_block = LL.block
                    load_val = self.load_store(LL, base_ptr, state[ispec.rf0][i])
                if self.mode == 'load':
                    # this dominates the final block
                    f = LL.phi(f32)
                    f.add_incoming(self.oob_value, before_cond)
                    f.add_incoming(load_val, if_block)
                    fs.append(f)
            t_block = LL.block
        if self.mode == 'load':
            fs_phi = [LL.phi(f32) for _ in range(ispec.ilp)]
            for i in range(ispec.ilp):
                fs_phi[i].add_incoming(self.oob_value, beforeall_block)
                fs_phi[i].add_incoming(fs[i], t_block)
            state[ispec.rf1] = fs_phi


class LoadGlobalF32(_LoadStoreGlobalBase):
    source_dtype = f32
    mode = 'load'


class StoreGlobalF32(_LoadStoreGlobalBase):
    source_dtype = f32
    mode = 'store'
