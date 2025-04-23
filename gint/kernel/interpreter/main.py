from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder


def build_main_loop(builder: PlatformIRBuilder):
    builder.printf(
        "Hello World %d! bidx %d tidx %d ws %d lid %d %s\n",
        builder.arg(0),
        builder.block_idx_x(),
        builder.thread_idx_x(),
        builder.warp_size(),
        builder.lane_id(),
        builder.string_literal("mooo")
    )
    builder.ret_void()


def build_interpreter_main_nvptx() -> ir.Module:
    mod = ir.Module('gint_device_module')
    mod.triple = "nvptx64-nvidia-cuda"
    mod.data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
    geval = ir.Function(mod, ir.FunctionType(void, [i32]), "geval")
    geval.calling_convention = "ptx_kernel"
    entry_bb = geval.append_basic_block("entry")
    builder = NVPTXIRBuilder(entry_bb)
    build_main_loop(builder)
    return mod
