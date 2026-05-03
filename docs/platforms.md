# Platform / GPU support

Reference for backend-specific details.

## Host-Side Executor

- `gint/host/executor.py`: core program execution interface + `get_executor()` auto-detection. `_convert_arg` fast-paths `torch.Tensor` (the dominant input type) by reading `data_ptr/shape/stride` directly instead of going through `__cuda_array_interface__` (~4× faster per arg). Other CUDA-array-interface producers (cupy, numba) still use the generic CAI parser. `BaseExecutableProgram.__call__` is wrapped with a guarded `torch.compiler.disable` that only fires when `torch.compiler.is_compiling()` is true — eliminating the ~2 µs/call dynamo-frame overhead on hot inference paths.
- `gint/host/cuda/executor_impl.py`: NVIDIA-specific implementation (CudaExecutor). Loads kernel from `gint/host/cuda/gint.fatbin.xz` (not tracked in git — must be generated via `./generate.sh`). `execute()` is split into a hot path and `_build_cacheline()`. The cacheline pre-builds the `(c_void_p * 5)()` `kernel_args` array that `cuLaunchKernel` needs — `dcode`/`dinfo` device pointers and `nargs`/`grid_dim`/`flag` ctypes wrappers are all per-shape constants, so the per-call hot path skips `prepare_arg`+`CTypesWrapper` and goes straight to `cuLaunchKernel`. Cacheline also stores the resolved `cufunc`, computed `(gx, gy, gz, bx, by, bz)` launch dims, and `smem_bytes`. The `_keepalive` tuple holds references to the c_int wrappers so the addresses inside `kernel_args` stay valid.
- `gint/host/hip/executor_impl.py`: AMD-specific implementation (HipExecutor). Loads kernel from `gint/host/hip/gint.fatbin.xz` (not tracked in git — must be generated via `./generate_amdgcn.sh`).
- **Backend selection**: `GINT_BACKEND` env var (`"cuda"` or `"hip"`) for explicit override; default tries CUDA first, falls back to HIP.
- `gint/host/utils.py`: shared utilities (`fill_tensor_info`, `cdiv`) used by both executors.
- **`execute_indirect(programs, args_list, indices)`**: launches multiple different programs in a single kernel call.
  - `programs`: list of `BaseExecutableProgram` instances.
  - `args_list`: list of tensor arg tuples, one per program (must match `programs` length).
  - `indices`: int array of size `grid_dim`, mapping each warp to a program index.
  - Builds per-program device code and tensor info, then assembles per-warp pointer tables for the kernel's indirect mode (`flag=1`).
  - Tensor info grid dimensions interact with `logical_program_idx()` via `urem`, so contiguous warp assignments per program work naturally.

## Kernel Specialization

- Per-vendor implementations in `gint/kernel/platforms/`:
  - `nvptx.py`: NVIDIA — `llvm.nvvm.*` intrinsics, `__nv_*` libdevice math, `nvptx64-nvidia-cuda` triple.
  - `amdgcn.py`: AMD RDNA3+ — `llvm.amdgcn.*` intrinsics, `__ocml_*_f32` math, `amdgcn-amd-amdhsa` triple.
- **llvmlite limitation**: `FunctionAttributes.add()` doesn't support key-value attributes (e.g., `"amdgpu-flat-work-group-size"="32,128"`). The AMDGCN builder works around this by overriding `emit()` to inject LLVM attribute groups via regex post-processing of the serialized IR text.

## cuda-python / cuda-bindings Version Compatibility

The `cuda-python` package was restructured into a metapackage starting at v12.6. The actual bindings live in `cuda-bindings`. Both 12.x (up to 12.9.x) and 13.x lines are actively maintained in parallel.

| Version line | Import style | Status |
|---|---|---|
| cuda-python <=12.5 | `from cuda import cuda` (old trampoline) | EOL |
| cuda-bindings 12.6–12.9.x | `import cuda.bindings.driver as cuda` | Maintained |
| cuda-bindings 13.0+ | `import cuda.bindings.driver as cuda` | Current |

**This project uses `import cuda.bindings.driver as cuda`**, which works on both 12.6+ and 13.x. The dependency spec `cuda-bindings>=12.6` accepts either line.

Key 13.0 changes: old trampoline modules (`cuda.cuda`, `cuda.cudart`) removed; GIL released for all C API calls; `int(cuda_obj)` deprecated in favor of `cuda.bindings.utils.get_cuda_native_handle()` (13.0+ only, `int()` still works). We use `int()` in `driver.py` and `executor_impl.py` — no change needed until `int()` is actually removed.

## NVIDIA GPU Architecture Support (Fatbin)

The fatbin (`gint/host/cuda/gint.fatbin.xz`) includes native SASS for:

| Architecture | Compute Capability | GPUs |
|---|---|---|
| Turing | sm_75 | RTX 2080, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090, L40 |
| Hopper | sm_90 | H100, H200 |
| Blackwell (datacenter) | sm_100 | B100, B200, GB200 |
| Blackwell (consumer) | sm_120 | RTX 5090, RTX 5080 |

Plus embedded `compute_120` PTX for forward compatibility with future architectures.

**Important**: sm_100 and sm_120 are **sibling** architectures, not parent/child. sm_120 is NOT a superset of sm_100 — they have different tensor core models. Both need explicit `-gencode` entries. PTX forward compatibility from `compute_90` is unreliable on Blackwell (the driver JIT can reject older PTX despite it being theoretically supported). Native SASS via CUDA 12.8+ is required.

**Build requirement**: `generate.sh` requires **CUDA Toolkit 12.8+** (for sm_100/sm_120 support). Earlier nvcc versions will fail on the Blackwell gencode flags.

## AMD GPU Architecture Support (HIP)

The AMDGCN fat binary (`gint/host/hip/gint.fatbin.xz`) bundles per-target HSACO code objects for:

| Architecture | GFX Target | GPUs |
|---|---|---|
| RDNA3 (discrete) | gfx1100, gfx1101 | RX 7900 XTX, RX 7800 XT, RX 7600 |
| RDNA4 | gfx1200, gfx1201 | RX 9070 XT, RX 9070 |

The fat binary uses the `__CLANG_OFFLOAD_BUNDLE__` format produced by `clang-offload-bundler`. `hipModuleLoadData` auto-selects the correct code object for the current GPU at runtime — no manual arch lookup needed.

**Build requirement**: `generate_amdgcn.sh` requires ROCm 7.0+ with OCML/OCKL bitcode libraries.

**CDNA not supported**: MI250/MI300 use wave64 (64-thread wavefronts), which would require 6-round reductions instead of 5, different shuffle masks, and changes to `lane_id` bounds. Only RDNA3+ (wave32) is supported.

## Cross-Platform Porting

The architecture cleanly separates platform-specific code into two layers: `PlatformIRBuilder` (device-side, in `gint/kernel/platforms/`) and `BaseExecutor` (host-side, in `gint/host/`). All 106 opcodes and the entire interpreter dispatch loop are platform-agnostic LLVM IR. A new platform needs:

1. **`PlatformIRBuilder` subclass** — implement ~17 abstract methods: thread/warp indexing, warp shuffle/broadcast/reduce, shared memory address space, transcendental math library calls, kernel module creation (triple, data layout, calling convention).
2. **`BaseExecutor` subclass** — device memory management, kernel loading, launch API.
3. **Build pipeline** — LLVM IR → platform binary (fatbin, HSACO, SPIR-V, etc.).

| Target | Difficulty | Notes |
|---|---|---|
| **AMD ROCm (HIP)** | **Implemented** | RDNA3+ (wave32) fully supported. `AMDGCNIRBuilder` in `gint/kernel/platforms/amdgcn.py`, `HipExecutor` in `gint/host/hip/`. Uses `llvm.amdgcn.*` intrinsics, `ocml` for math, `hip-python` for host API, `ds.bpermute` for warp reductions, `readlane` for broadcasts. CDNA (MI250/MI300, wave64) is NOT supported — would need 6-round reductions and different shuffle masks. |
| **OpenCL SPIR-V** | Moderate-Hard | Best cross-vendor path. `llvm-spirv` translates LLVM IR → OpenCL SPIR-V, and the OpenCL execution model (kernel functions with pointer args, flat address space, raw pointer arithmetic) matches our design closely. Subgroup ops (`cl_khr_subgroups`) map to warp primitives. Runs on Intel, AMD, NVIDIA via their OpenCL drivers. `pyopencl` for host API. Main issues: subgroup size varies by vendor (32/64/8-32), driver optimization quality lags CUDA. |
| **Vulkan SPIR-V** | Hard | Vulkan uses a fundamentally different execution model: no raw pointers (descriptor sets + buffer offsets), `GLCompute` execution model vs `Kernel`. Our pointer-table indirect dispatch doesn't translate directly. Would likely need a parallel SPIR-V codegen path rather than reusing llvmlite → `llvm-spirv`. Subgroup ops available (Vulkan 1.1+) but size varies. Math via `GLSL.std.450` extended instructions. |
| **Apple Metal GPU** | Hard | No LLVM IR → Metal path. Would need to generate MSL source or Metal IR directly. Metal has `simdgroup` ops (size 32 on Apple Silicon) that map well to our warp primitives. `pyobjc` for host API. Apple's `metal` compiler is clang-based but Metal IR format is undocumented. |
| **Apple NPU (ANE)** | Impossible | Not a programmable compute device — executes compiled neural network graphs (Core ML), no user-accessible ISA, no thread model. |

**Key constraint across all targets**: the interpreter assumes `WARP_SIZE=32` (32 threads per subgroup). RDNA3 AMD GPUs support wave32 natively, but CDNA and some Intel GPUs use 64 or variable sizes. This affects `lane_id` bounds, shuffle masks, and the tree reduction in `warp_allreduce` (5 rounds for 32, 6 for 64).
