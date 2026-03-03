from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ...platforms.common import *


BlockTensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, MAX_N_TENSORS),  # ptr with block offset

    ir.ArrayType(i32, MAX_N_TENSORS),  # thread stride
    ir.ArrayType(i32, MAX_N_TENSORS),  # width stride
    ir.ArrayType(i32, MAX_N_TENSORS),  # offset stride

    ir.ArrayType(i32, MAX_N_TENSORS),  # constraint 1 size
    ir.ArrayType(i32, MAX_N_TENSORS),  # constraint 2 size
    ir.ArrayType(ir.VectorType(i16, 4), MAX_N_TENSORS),  # constraint 1 width/thread/offset weight log2 packed
    ir.ArrayType(ir.VectorType(i16, 4), MAX_N_TENSORS),  # constraint 2 width/thread/offset weight log2 packed
])


def emit_load_tensor_infos(LL: PlatformIRBuilder, state: StackMachineState):
    lane_id = LL.lane_id()
    with LL.if_then(LL.icmp_signed('<', lane_id, LL.arg(2))):
        (base_ptr, elm_size,
         b_stride, b_size, b2t_stride, b2t_size, b2w_stride, b2w_size,
         t_stride, w_stride, o_stride,
         c1_size, c2_size,
         c1_w, c2_w
        )  = [
            LL.load(LL.gep(LL.arg(1), [i32(0), i32(eid), lane_id], inbounds=True))
            for eid in range(15)
        ]
        bidx = LL.logical_program_idx()
        rbidx = bidx
        b2t_idx = LL.urem(rbidx, b2t_size)
        rbidx = LL.udiv(rbidx, b2t_size)
        b2w_idx = LL.urem(rbidx, b2w_size)
        rbidx = LL.udiv(rbidx, b2w_size)
        b2t_total = LL.mul(b2t_idx, b2t_stride)
        b2w_total = LL.mul(b2w_idx, b2w_stride)
        batch_offset = LL.add(LL.mul(b2t_total, t_stride), LL.mul(b2w_total, w_stride))
        for i in range(4):  # batch dims
            bs = LL.extract_element(b_stride, i32(i))
            sz = LL.extract_element(b_size, i32(i))
            idx = LL.urem(rbidx, sz)
            rbidx = LL.udiv(rbidx, sz)
            batch_offset = LL.add(batch_offset, LL.mul(bs, idx))
        base_ptr = LL.gep(base_ptr, [LL.mul(LL.zext(batch_offset, i64), LL.zext(elm_size, i64))], inbounds=True)
        
        c1_ww, c1_wt = LL.extract_element(c1_w, i32(0)), LL.extract_element(c1_w, i32(1))
        c1_size = LL.sub(c1_size, LL.add(LL.mul(b2t_total, LL.zext(c1_wt, i32)), LL.mul(b2w_total, LL.zext(c1_ww, i32))))
        c2_ww, c2_wt = LL.extract_element(c2_w, i32(0)), LL.extract_element(c2_w, i32(1))
        c2_size = LL.sub(c2_size, LL.add(LL.mul(b2t_total, LL.zext(c2_wt, i32)), LL.mul(b2w_total, LL.zext(c2_ww, i32))))

        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        for eid, val in enumerate([
            base_ptr, t_stride, w_stride, o_stride,
            c1_size, c2_size,
            c1_w, c2_w
        ]):
            LL.store(val, LL.gep(smem_base, [i32(0), i32(eid), lane_id], inbounds=True))
    LL.warp_sync()
