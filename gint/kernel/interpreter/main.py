from llvmlite import ir
from ..platforms.common import *
from ..platforms.platform import PlatformIRBuilder
from ..platforms.nvptx import NVPTXIRBuilder


def build_main_loop(builder: PlatformIRBuilder):
    builder.ret_void()


def build_interpreter_main_nvptx() -> ir.Module:
    mod = ir.Module('gint_device_module')
    mod.triple = "nvptx64-nvidia-cuda"
    geval = ir.Function(mod, ir.FunctionType(void, [i32]), "geval")
    geval.calling_convention = "ptx_kernel"
    entry_bb = geval.append_basic_block("entry")
    builder = NVPTXIRBuilder(entry_bb)
    build_main_loop(builder)
    return mod
