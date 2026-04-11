
def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def fill_tensor_info(ti, input_infos, slot_offset: int = 0):
    for i, t in enumerate(input_infos):
        si = slot_offset + i
        ti.elm_size[si] = t.elm_size
        rev_bs = t.batch_strides[::-1]
        rev_bsz = t.batch_shape[::-1]
        assert len(rev_bs) == len(rev_bsz) <= 4, "At most 4 block axes supported!"
        for j in range(4):
            if j < len(rev_bsz):
                ti.batch_strides[si][j] = rev_bs[j]
                ti.batch_shape[si][j] = rev_bsz[j]
            else:
                ti.batch_strides[si][j] = 0
                ti.batch_shape[si][j] = 1
        ti.block_shape_stride_1[si][0] = t.block_shape_stride_1[0]
        ti.block_shape_stride_1[si][1] = t.block_shape_stride_1[1]
        ti.block_shape_stride_2[si][0] = t.block_shape_stride_2[0]
        ti.block_shape_stride_2[si][1] = t.block_shape_stride_2[1]
        ti.block_grid_dims[si][0] = t.block_grid_dims[0]
        ti.block_grid_dims[si][1] = t.block_grid_dims[1]
        ti.block_grid_steps[si][0] = t.block_grid_steps[0]
        ti.block_grid_steps[si][1] = t.block_grid_steps[1]
