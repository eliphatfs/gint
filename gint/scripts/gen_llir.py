import rich
import argparse
import subprocess
from typing import Literal, Optional
from gint.kernel.interpreter.main import build_interpreter_main_nvptx


def invoke_clang_shim(llir: bytes, target: Literal['ptx'] = 'ptx', cc: Optional[int] = None, emit_llir: bool = False):
    targets = {'ptx': 'nvptx64-nvidia-cuda'}
    if target not in targets:
        raise ValueError("Unsupported target", target, "supported targets", targets)
    return subprocess.check_output([
        'clang',
        '--target=%s' % targets[target],
        '-march=sm_%d' % cc if cc else '',
        '-x', 'ir',
        '-S',
        '-O3',
        '-emit-llvm' if emit_llir else '',
        '-o', '-',
        '-'
    ], input=llir)


def invoke_opt_shim(llir: bytes):
    return subprocess.check_output([
        'opt',
        '-passes=verify',
        '-S',
        '-'
    ], input=llir)
    


def main():
    argp = argparse.ArgumentParser('gint-gen-llir')
    argp.add_argument("-o", "--output-path", type=str, default=None)
    argp.add_argument("-t", "--target", type=str, default="llir", choices=['llir', 'ptx'])
    argp.add_argument("-c", "--check-only", action='store_true', help='Check the original LLIR only and do not compile.')
    argp.add_argument(
        "-E", "--emit-llir",
        action='store_true',
        help="Emit optimized LLIR for the target (not effective when target is 'llir' when the program generates unoptimized LLIR)."
    )
    argp.add_argument("--cc", type=int, default=None)
    args = argp.parse_args()
    
    mod = build_interpreter_main_nvptx()
    ir = mod.emit()
    if args.check_only:
        ir = invoke_opt_shim(ir)
    elif args.target != 'llir':
        ir = invoke_clang_shim(ir, args.target, args.cc, args.emit_llir)
    if args.output_path is None:
        rich.print(ir.decode().strip().replace('[', '\\['))
    else:
        with open(args.output_path, "wb") as fo:
            fo.write(ir)
