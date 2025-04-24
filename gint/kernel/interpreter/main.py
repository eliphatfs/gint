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
    builder = NVPTXIRBuilder.create_kernel_module(ir.FunctionType(void, [i32]), "geval")
    build_main_loop(builder)
    return builder.module
