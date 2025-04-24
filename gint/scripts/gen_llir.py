import argparse
import subprocess
from typing import Literal, Optional
from gint.kernel.interpreter.main import build_interpreter_main_nvptx


def invoke_clang_shim(llir: bytes, target: Literal['ptx'] = 'ptx', cc: Optional[int] = None):
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
        '-o', '-',
        '-'
    ], input=llir)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-o", "--output-path", type=str, required=True)
    argp.add_argument("-t", "--target", type=str, default="llir", choices=['llir', 'ptx'])
    argp.add_argument("--cc", type=int, default=None)
    args = argp.parse_args()
    
    mod = build_interpreter_main_nvptx()
    ir = str(mod).encode()
    if args.target != 'llir':
        ir = invoke_clang_shim(ir, args.target, args.cc)
    with open(args.output_path, "wb") as fo:
        fo.write(ir)
