from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ...platforms.common import *


BlockTensorInfo = ir.LiteralStructType([
    ir.ArrayType(p_i8, MAX_N_TENSORS),  # ptr with block offset
    ir.ArrayType(ir.VectorType(i32, 2), MAX_N_TENSORS),  # block shape and stride 1
    ir.ArrayType(ir.VectorType(i32, 2), MAX_N_TENSORS),  # block shape and stride 2
    ir.ArrayType(i32, MAX_N_TENSORS),  # advanced offset
])


def emit_load_tensor_infos(LL: PlatformIRBuilder, state: StackMachineState):
    lane_id = LL.lane_id()
    with LL.if_then(LL.icmp_signed('<', lane_id, LL.arg(2))):
        (base_ptr, elm_size, batch_strides, batch_shape, block_s1, block_s2, block_dims, block_steps)  = [
            LL.load(LL.gep(LL.arg(1), [i32(0), i32(eid), lane_id], inbounds=True))
            for eid in range(8)
        ]
        bidx = LL.logical_program_idx()
        rbidx = bidx
        b1_shape, b1_stride = LL.extract_element(block_s1, i32(0)), LL.extract_element(block_s1, i32(1))
        b2_shape, b2_stride = LL.extract_element(block_s2, i32(0)), LL.extract_element(block_s2, i32(1))
        block_dim_1, block_dim_2 = LL.extract_element(block_dims, i32(0)), LL.extract_element(block_dims, i32(1))
        block_step_1, block_step_2 = LL.extract_element(block_steps, i32(0)), LL.extract_element(block_steps, i32(1))
        b1_idx = LL.urem(rbidx, block_dim_1)
        rbidx = LL.udiv(rbidx, block_dim_1)
        b2_idx = LL.urem(rbidx, block_dim_2)
        rbidx = LL.udiv(rbidx, block_dim_2)
        batch_offset = LL.add(LL.mul(LL.mul(b1_idx, block_step_1), b1_stride), LL.mul(LL.mul(b2_idx, block_step_2), b2_stride))
        for i in range(4):  # batch dims
            bs = LL.extract_element(batch_strides, i32(i))
            sz = LL.extract_element(batch_shape, i32(i))
            idx = LL.urem(rbidx, sz)
            rbidx = LL.udiv(rbidx, sz)
            batch_offset = LL.add(batch_offset, LL.mul(bs, idx))
        base_ptr = LL.gep(base_ptr, [LL.mul(LL.zext(batch_offset, i64), LL.zext(elm_size, i64))], inbounds=True)
        
        b1_shape = LL.sub(b1_shape, LL.mul(b1_idx, block_step_1))
        b2_shape = LL.sub(b2_shape, LL.mul(b2_idx, block_step_2))

        block_s1 = LL.insert_element(block_s1, b1_shape, i32(0))
        block_s2 = LL.insert_element(block_s2, b2_shape, i32(0))

        smem_base = state.smem_base
        smem_base = LL.bitcast(smem_base, BlockTensorInfo.as_pointer(LL.smem_addrspace()))
        for eid, val in enumerate([
            base_ptr, block_s1, block_s2, i32(0)
        ]):
            LL.store(val, LL.gep(smem_base, [i32(0), i32(eid), lane_id], inbounds=True))
    LL.warp_sync()
