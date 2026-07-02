import os
import re
import rich
import argparse
import subprocess
from typing import Literal, Optional
from gint.kernel.interpreter.main import build_interpreter_main_nvptx, build_interpreter_main_amdgcn
from gint.kernel import platforms


# Attribute set forcing ocml math functions to inline. ocml ships asin/acos/
# atan/atan2/erf/pow (and helpers fmuladd/atanred/epln/expep/exp) marked
# `convergent` and without amdgpu-no-* attributes, so LLVM keeps them as
# out-of-line s_swappc calls. A call is a register live-out point: the dispatch
# switch fans out into per-stack-depth case blocks that each keep their own
# stack VGPRs live across the call, so the union pins VGPR at ~255 and forces
# spills. These functions are just polynomial approximations (fma + hardware
# sqrt/log/exp), so always-inline + the amdgpu-no-* set lets LLVM inline them
# fully -> 0 swappc, VGPR 256->~140, zero spill, occupancy doubled.
_AMDGPU_NO_ATTRS = (
    'mustprogress nofree norecurse nosync nounwind willreturn memory(none)'
    ' "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue"'
    ' "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init"'
    ' "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr"'
    ' "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr"'
    ' "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z"'
    ' "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z"'
    ' "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true"'
    ' "stack-protector-buffer-size"="8" "uniform-work-group-size"="false"'
)
_INLINE_ATTRS = '{ alwaysinline ' + _AMDGPU_NO_ATTRS + ' }'


def _amdgcn_inline_ocml_math(llir_bc: bytes) -> bytes:
    """Disassemble the linked bitcode, force ocml math functions to
    always-inline (dropping convergent, adding amdgpu-no-*), reassemble."""
    text = subprocess.check_output(
        ['llvm-dis', '-o', '-'], input=llir_bc).decode()
    # Find every attribute group used by an ocml/ocmlpriv math function and
    # replace it with the inline-forcing set. Collect the group numbers first.
    import re
    grps = set()
    for m in re.finditer(
            r'define[^@]*@(__ocml[a-z_]*_f\d+|__ocmlpriv_[a-z_]*_f\d+)[^#]*#(\d+)',
            text):
        grps.add(m.group(2))
    for g in sorted(grps):
        pat = r'attributes #' + g + r' = \{[^}]*\}'
        text = re.sub(pat, 'attributes #' + g + ' = ' + _INLINE_ATTRS, text)
    return subprocess.check_output(['llvm-as', '-o', '-'], input=text.encode())


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
            os.path.join(bc_dir, 'oclc_correctly_rounded_sqrt_off.bc'),
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
        # Force ocml math functions (asin/acos/atan/atan2/erf/pow/exp and
        # their helpers fmuladd/atanred/epln/expep) to inline into geval.
        # ocml ships them marked `convergent` and without amdgpu-no-* attributes,
        # so LLVM keeps them as out-of-line s_swappc calls; a call is a register
        # live-out point that pins the dispatch loop's stack VGPRs at ~255 and
        # forces spills. These functions are just polynomial approximations
        # (fma + hardware sqrt/log/exp), so always-inline + the amdgpu-no-* set
        # lets LLVM inline them fully -> 0 swappc, VGPR 256->~140, zero spill,
        # occupancy doubled. (Only for amdgcn; NVPTX libdevice uses nvvm_reflect
        # and is handled separately.)
        if target == 'amdgcn':
            llir = _amdgcn_inline_ocml_math(llir)
        # Use llc directly (bypassing clang's integrated backend) so that a
        # patched llc with the s_setpc jump-table dispatch lowering can be used.
        # clang's integrated backend would ignore the patched llc on PATH.
        if emit_llir:
            return llir
        cmd = [
            'llc',
            '-mtriple=%s' % targets[target],
            '-mcpu=%s' % (gfx or 'gfx1100'),
            '-O3',
            '-o', '-',
            '-',
        ]
        # Allow disabling jump tables (bit-test/comparison-tree dispatch
        # instead) for diagnosis via the GINT_NO_JT env var.
        if os.environ.get('GINT_NO_JT'):
            cmd.insert(1, '-max-jump-table-size=0')
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
    args = argp.parse_args()

    if args.target == 'amdgcn':
        mod = build_interpreter_main_amdgcn(args.gfx)
    else:
        mod = build_interpreter_main_nvptx()
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
