import os
import re
import rich
import argparse
import subprocess
from typing import Literal, Optional
from gint.kernel.interpreter.main import DispatchMode, build_interpreter_main_nvptx, build_interpreter_main_amdgcn
from gint.kernel import platforms


def invoke_clang_shim(llir: bytes, target: Literal['ptx', 'amdgcn'] = 'ptx', cc: Optional[int] = None, gfx: Optional[str] = None, emit_llir: bool = False):
    targets = {
        'ptx': 'nvptx64-nvidia-cuda',
        'amdgcn': 'amdgcn-amd-amdhsa',
    }
    if target not in targets:
        raise ValueError("Unsupported target", target, "supported targets", targets)
    if target == 'ptx':
        libdevice = os.path.join(os.path.dirname(os.path.abspath(platforms.__file__)), 'nvptx.libdevice.10.bc')
        llir = subprocess.check_output([
            'llvm-link',
            '--only-needed',
            '-', libdevice,
            '-o', '-'
        ], input=llir)
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
    elif target == 'amdgcn':
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        bc_dir = os.path.join(rocm_path, 'amdgcn', 'bitcode')
        # Extract ISA version number from gfx name for oclc_isa_version bitcode
        isa_match = re.match(r'gfx(\d+)', gfx or 'gfx1100')
        isa_version = isa_match.group(1) if isa_match else '1100'
        bc_files = [
            os.path.join(bc_dir, 'ocml.bc'),
            os.path.join(bc_dir, 'ockl.bc'),
            os.path.join(bc_dir, 'oclc_wavefrontsize64_off.bc'),
            os.path.join(bc_dir, f'oclc_isa_version_{isa_version}.bc'),
            os.path.join(bc_dir, 'oclc_daz_opt_off.bc'),
            os.path.join(bc_dir, 'oclc_finite_only_off.bc'),
            os.path.join(bc_dir, 'oclc_unsafe_math_off.bc'),
        ]
        # Filter to only existing bitcode files (some may not be present)
        bc_existing = [f for f in bc_files if os.path.exists(f)]
        if not bc_existing:
            raise FileNotFoundError(
                f"No OCML/OCKL bitcode found in {bc_dir}. "
                f"Set ROCM_PATH to your ROCm installation."
            )
        llir = subprocess.check_output(
            ['llvm-link', '--only-needed', '-'] + bc_existing + ['-o', '-'],
            input=llir
        )
        cmd = [
            'clang',
            '--target=%s' % targets[target],
            '-mcpu=%s' % (gfx or 'gfx1100'),
            '-S',
            '-x', 'ir',
            '-O3',
        ]
        if emit_llir:
            cmd += ['-emit-llvm']
        cmd += ['-o', '-', '-']
        return subprocess.check_output(cmd, input=llir)


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
    argp.add_argument("-t", "--target", type=str, default="llir", choices=['llir', 'ptx', 'amdgcn'])
    argp.add_argument("-c", "--check-only", action='store_true', help='Check the original LLIR only and do not compile.')
    argp.add_argument(
        "-E", "--emit-llir",
        action='store_true',
        help="Emit optimized LLIR for the target (not effective when target is 'llir' when the program generates unoptimized LLIR)."
    )
    argp.add_argument("--cc", type=int, default=None)
    argp.add_argument("--gfx", type=str, default="gfx1100", help="AMD GFX target (e.g. gfx1100, gfx11-generic)")
    argp.add_argument("--dispatch-mode", type=str, default="switch",
                      choices=['switch', 'alloca-switch', 'balanced', 'optimal', 'alloca-balanced'],
                      help="Dispatch mode for opcode dispatch (default: switch)")
    args = argp.parse_args()

    dispatch_mode = {
        'switch': DispatchMode.SWITCH,
        'alloca-switch': DispatchMode.ALLOCA_SWITCH,
        'balanced': DispatchMode.BALANCED_TREE,
        'optimal': DispatchMode.OPTIMAL_TREE,
        'alloca-balanced': DispatchMode.ALLOCA_BALANCED,
    }[args.dispatch_mode]

    if args.target == 'amdgcn':
        mod = build_interpreter_main_amdgcn(args.gfx, dispatch_mode=dispatch_mode)
    else:
        mod = build_interpreter_main_nvptx(dispatch_mode=dispatch_mode)
    ir = mod.emit()
    if args.check_only:
        ir = invoke_opt_shim(ir)
    elif args.target != 'llir':
        ir = invoke_clang_shim(ir, args.target, args.cc, args.gfx, args.emit_llir)
    if args.output_path is None:
        rich.print(ir.decode().strip().replace('[', '\\['))
    else:
        with open(args.output_path, "wb") as fo:
            fo.write(ir)
