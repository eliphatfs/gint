from llvmlite import ir

from .common import *
from .platform import PlatformIRBuilder


class AMDGCNIRBuilder(PlatformIRBuilder):

    _gfx: str = "gfx1100"
    _is_amdgcn: bool = True

    @classmethod
    def create_kernel_module(cls, fn_type: ir.FunctionType, fn_name: str, gfx: str = "gfx1100"):
        mod = ir.Module('gint_device_module')
        mod.triple = "amdgcn-amd-amdhsa"
        mod.data_layout = (
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
            "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32"
            "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
            "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
            "-n32:64-S32-A5-G1-ni:7:8:9"
        )
        geval = ir.Function(mod, fn_type, fn_name)
        geval.calling_convention = "amdgpu_kernel"
        entry_bb = geval.append_basic_block("entry")
        builder = AMDGCNIRBuilder(entry_bb)
        builder._gfx = gfx
        builder._kernel_fn_names = [fn_name]
        return builder

    @classmethod
    def add_kernel(cls, prev: 'AMDGCNIRBuilder', fn_type: ir.FunctionType, fn_name: str):
        """Append a second kernel function to the module of an existing builder
        and return a fresh builder positioned at its entry block. The new
        builder shares the kernel-name list, so emit() attaches the attribute
        group to every kernel."""
        mod = prev.module
        geval = ir.Function(mod, fn_type, fn_name)
        geval.calling_convention = "amdgpu_kernel"
        entry_bb = geval.append_basic_block("entry")
        builder = AMDGCNIRBuilder(entry_bb)
        builder._gfx = prev._gfx
        builder._kernel_fn_names = prev._kernel_fn_names
        builder._kernel_fn_names.append(fn_name)
        return builder

    def emit(self):
        # llvmlite doesn't support arbitrary key-value function attributes,
        # so we inject them via LLVM IR attribute groups after serialization.
        llir = super().emit().decode()
        import re
        for fn_name in self._kernel_fn_names:
            # Find the closing paren of the argument list in the define line and add #0
            # The pattern is: define amdgpu_kernel void @"<fn_name>"(...)\n{
            llir = re.sub(
                r'(define\s+amdgpu_kernel\s+void\s+@"' + re.escape(fn_name) + r'"\([^)]*\))',
                r'\1 #0',
                llir,
                count=1,
            )
        llir += (
            f'\nattributes #0 = {{'
            f' "amdgpu-flat-work-group-size"="32,128"'
            f' "target-cpu"="{self._gfx}"'
            f' }}\n'
        )
        return llir.encode()

    def smem_addrspace(self) -> int:
        return 3

    def warp_sync(self) -> None:
        self.intrinsic('llvm.amdgcn.wave.barrier', void, [])

    def thread_idx_x(self) -> ir.Value:
        return self.intrinsic('llvm.amdgcn.workitem.id.x', i32, [])

    def thread_idx_y(self) -> ir.Value:
        return self.intrinsic('llvm.amdgcn.workitem.id.y', i32, [])

    def logical_program_idx(self) -> ir.Value:
        # Read workgroup_size_y from HSA AQL dispatch packet (uint16 at byte offset 6)
        dispatch_ptr = self.intrinsic('llvm.amdgcn.dispatch.ptr', i8.as_pointer(4), [])
        wg_size_y_ptr = self.gep(dispatch_ptr, [i32(6)], inbounds=True)
        wg_size_y_ptr = self.bitcast(wg_size_y_ptr, i16.as_pointer(4))
        wg_size_y = self.zext(self.load(wg_size_y_ptr), i32)
        wg_id_x = self.intrinsic('llvm.amdgcn.workgroup.id.x', i32, [])
        wi_id_y = self.intrinsic('llvm.amdgcn.workitem.id.y', i32, [])
        return self.add(self.mul(wg_id_x, wg_size_y), wi_id_y)

    def warp_size(self) -> ir.Value:
        return i32(32)

    def lane_id(self) -> ir.Value:
        return self.intrinsic('llvm.amdgcn.mbcnt.lo', i32, [i32(-1), i32(0)])

    def printf(self, fmt: str, *args: ir.Value) -> ir.Value:
        # AMDGCN printf requires OCKL hostcall infrastructure — stub for now
        return i32(0)

    def warp_broadcast_lane(self, value: ir.NamedValue, lane: ir.Value) -> ir.Value:
        if value.type == i32:
            return self.intrinsic('llvm.amdgcn.readlane', i32, [value, lane])
        if value.type == f32:
            ival = self.bitcast(value, i32)
            result = self.intrinsic('llvm.amdgcn.readlane', i32, [ival, lane])
            return self.bitcast(result, f32)
        raise TypeError("Expected value type f32 or i32, got", value.type)

    def make_uniform(self, value: ir.NamedValue) -> ir.NamedValue:
        # Promote a per-lane (possibly divergent) value to a uniform (scalar)
        # value by taking the value from the first active lane. On AMDGPU this
        # is llvm.amdgcn.readfirstlane, which lets the dispatch-switch index be
        # uniform so the patched backend can lower it to an O(1) s_setpc jump
        # table (see docs/kernel.md dispatch).
        if value.type == i32:
            return self.intrinsic('llvm.amdgcn.readfirstlane', i32, [value])
        if value.type == f32:
            ival = self.bitcast(value, i32)
            return self.bitcast(
                self.intrinsic('llvm.amdgcn.readfirstlane', i32, [ival]), f32)
        raise TypeError("Expected value type f32 or i32, got", value.type)

    def warp_allreduce_f32(self, value: ir.Value, op: EReducePrimitiveOp) -> ir.Value:
        red = value
        lane = self.lane_id()
        for mask in [16, 8, 4, 2, 1]:
            target = self.xor(lane, i32(mask))
            byte_offset = self.shl(target, i32(2))
            ival = self.bitcast(red, i32)
            comm_i32 = self.intrinsic('llvm.amdgcn.ds.bpermute', i32, [byte_offset, ival])
            comm = self.bitcast(comm_i32, f32)
            if op == EReducePrimitiveOp.Sum:
                red = self.fadd(red, comm)
            elif op == EReducePrimitiveOp.Max:
                red = self.intrinsic('llvm.maximum.f32', f32, [red, comm])
            elif op == EReducePrimitiveOp.Min:
                red = self.intrinsic('llvm.minimum.f32', f32, [red, comm])
            elif op == EReducePrimitiveOp.Prod:
                red = self.fmul(red, comm)
        return red

    def special_unary(self, value: ir.Value, op: EUnarySpecialOp) -> ir.Value:
        intrinsic_map = {
            EUnarySpecialOp.Sqrt: '__ocml_sqrt_f32',
            EUnarySpecialOp.Sin: '__ocml_sin_f32',
            EUnarySpecialOp.Cos: '__ocml_cos_f32',
            EUnarySpecialOp.Tan: '__ocml_tan_f32',
            EUnarySpecialOp.ArcSin: '__ocml_asin_f32',
            EUnarySpecialOp.ArcCos: '__ocml_acos_f32',
            EUnarySpecialOp.ArcTan: '__ocml_atan_f32',
            EUnarySpecialOp.Exp: '__ocml_exp_f32',
            EUnarySpecialOp.Exp2: '__ocml_exp2_f32',
            EUnarySpecialOp.Log: '__ocml_log_f32',
            EUnarySpecialOp.Log2: '__ocml_log2_f32',
            EUnarySpecialOp.RSqrt: '__ocml_rsqrt_f32',
            EUnarySpecialOp.Erf: '__ocml_erf_f32',
        }
        return self.intrinsic(intrinsic_map[op], f32, [value])

    def special_binary(self, a: ir.Value, b: ir.Value, op: EBinarySpecialOp) -> ir.Value:
        intrinsic_map = {
            EBinarySpecialOp.ArcTan2: '__ocml_atan2_f32',
            EBinarySpecialOp.Pow: '__ocml_pow_f32',
        }
        return self.intrinsic(intrinsic_map[op], f32, [a, b])
