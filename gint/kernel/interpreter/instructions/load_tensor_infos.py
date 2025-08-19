from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ...platforms.common import *


BlockTensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, 8),  # ptr with block offset
    ir.ArrayType(i32, 8),  # ilp stride
    ir.ArrayType(i32, 8),  # ilp size
    ir.ArrayType(i32, 8),  # thread stride
    ir.ArrayType(i32, 8),  # thread size
    ir.ArrayType(i32, 8),  # ilp contribution stride to thread offset
])


def emit_load_tensor_infos(LL: PlatformIRBuilder, state: StackMachineState):
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
                sz = LL.udiv(LL.add(sz, i32(state.reg_width - 1)), i32(state.reg_width))  # ceil div
            idx = LL.urem(rbidx, sz)
            rbidx = LL.udiv(rbidx, sz)
            if i == 0:
                idx = LL.mul(idx, i32(state.reg_width))
                ilp_sz = LL.sub(ilp_sz, idx)
            base_ptr = LL.gep(base_ptr, [LL.mul(LL.mul(LL.zext(bs, i64), LL.zext(idx, i64)), LL.zext(elm_sz, i64))], inbounds=True)
            rt_ofs = LL.add(rt_ofs, LL.mul(idx, btofss))
        t_size = LL.sub(t_size, rt_ofs)
        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        for eid, val in enumerate([base_ptr, ilp_s, ilp_sz, t_stride, t_size, ilp_ofs_stride]):
            LL.store(val, LL.gep(smem_base, [i32(0), i32(eid), lane_id], inbounds=True))
    LL.warp_sync()
