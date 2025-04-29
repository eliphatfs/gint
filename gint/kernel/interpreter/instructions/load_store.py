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
            b_strides, b_sizes, t_stride, t_size, elm_sz, resv, base_ptr = [
                LL.load(LL.gep(LL.arg(1), [lane_id, i32(eid)], inbounds=True))
                for eid in range(7)
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
        ptr = LL.phi(p_i8)
        ptr.add_incoming(base_ptr, if_block)
        ptr.add_incoming(p_i8(None), before_block)
        state[ispec.rof][0] = ptr
        state[ispec.rss] = [ilps, ilpsz, rts, rtsz]

class LoadGlobal(Instruction):
    pass
