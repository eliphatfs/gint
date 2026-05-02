# Kernel / Interpreter details

Reference for the device-side bytecode interpreter. See `gint/kernel/interpreter/` for source.

## Stack-Based Interpreter

- **Micro-architecture**:
  - `REG_WIDTH = 4`: each lane processes 4 elements simultaneously (ILP).
  - Pool/stack/register sizes are per-variant (see Kernel Variants in CLAUDE.md). Stack grows upward from index 0; virtual registers occupy the top `NUM_REGS` slots counting downward (`reg n = pool[POOL_SIZE-1-n]`).
  - Dispatch: large switch-case in LLVM IR with profile weights.
  - Uses PHI nodes for state persistence across dispatches (one dispatch block per stack depth `0..MAX_STACK`).

## Kernel Variants

- The fatbin/HSACO ships **two kernel variants** under symbols `geval_s7` and `geval_l12`. Both implement the full interpreter; they differ only in pool/stack/reg sizes:
  - `s7`: `POOL_SIZE=7, NUM_REGS=4, MAX_STACK=7` — fits all pointwise/streaming workloads (max measured: stack=6, regs=0). Smaller register pressure on the GPU → better occupancy.
  - `l12`: `POOL_SIZE=12, NUM_REGS=8, MAX_STACK=8` — required by register-heavy kernels (e.g. `inv4x4` uses all 8 regs).
- **Variant table**: `gint.kernel.interpreter.main.VARIANTS` maps name → `(pool_size, num_regs, max_stack)`. `variant_kernel_name(name)` formats the kernel symbol (`geval_<name>`).
- **IR generation**: `build_interpreter_main_nvptx()` / `build_interpreter_main_amdgcn()` emit ONE module containing every variant's kernel. The `dynamic_smem` shared global is created once and shared across kernels. AMDGCN's `emit()` post-processor attaches the `#0` attribute group to every kernel symbol.
- **Per-variant opcode filtering**: `FLoadRegN`/`FStoreRegN` opcodes for `N >= num_regs` are dropped from the dispatch table of smaller variants (otherwise they would alias into normal stack slots via `pool[pool_size-1-N]`). The `s7` variant only accepts opcodes 87–90 (load reg 0–3) and 95–98 (store reg 0–3).
- **Host-side selection** (`gint/host/executor.py:select_variant`): runs `analyze_bytecode` on each compiled program, picks the smallest variant whose `(num_regs, max_stack)` covers the program. Accepts either raw bytecode or a `ProgramData` (preferred — reuses the lazily-cached `pd.stats`, since `analyze_bytecode` walks the whole stream and is the dominant cost on the cache-miss path). Variant decision is cached alongside the per-program device buffers in `program._cu` / `program._hip`.
- **`execute_indirect`**: picks `max(variant) over all programs in the batch` (one launch per call, not bucketed). Mixed batches are rare; bucketing would multiply launch overhead.
- **Build/deploy**: `./generate.sh` and `./generate_amdgcn.sh` bundle BOTH variants into one fatbin/HSACO.

## Virtual Register File

- 8 virtual registers (`reg 0`–`reg 7`) backed by the top 8 slots of the unified pool.
- `FLoadRegN` / `FStoreRegN` (N=0..7): specialized per-register instructions with **zero select/compare overhead** — each directly aliases `pool[pool_size-1-N]`.
- Generic `FLoadReg` / `FStoreReg` with a runtime operand were replaced by 16 specialized opcodes (87–102) to eliminate register pressure from select chains.
- Frontend: `fload_reg(n)` / `fstore_reg(n)` dispatch to the correct specialized class via `LOAD_REGS[n]` / `STORE_REGS[n]`.

## Width-Lane Permutation and Shuffle

### `binary_cond_tree` utility (`move.py`)
- Module-level function (not a method) used by `DupBroadcastW`, `FPermW`, and `FShuf2`.
- Builds a binary selection tree over a list of `ir.Value`s, selecting one based on a runtime i32 index.

### `FPermW` (opcode 104)
- Permutes the 4 width-lanes of the top-of-stack vector in-place.
- Operand is an i32 encoded as **i8x4 little-endian**: bits 7..0 = source lane for output 0, bits 15..8 = source lane for output 1, etc.
- Frontend: `fperm_w(i0, i1, i2, i3)` encodes the four indices and emits opcode 104.
- **Broadcast idiom**: `fperm_w(i, i, i, i)` broadcasts lane i in-place — preferred over `dup_broadcast_w(i); swap(); pop()` (1 instruction vs 3).

### `FShuf2` (opcode 105)
- Two-source shuffle: `VecShuffle(vec1, vec2, x, y, z, w)` → `(vec1[x], vec1[y], vec2[z], vec2[w])`.
- vec2 is on top of stack, vec1 is below; pops both, pushes shuffled result.
- Same i8x4 little-endian operand encoding as `FPermW`; indices x,y select from vec1, z,w from vec2.
- Frontend: `fshuf2(x, y, z, w)` encodes indices and emits opcode 105.

## Indirect Load/Store (Scatter/Gather)

- `LoadGlobal1DF32Indirect` / `StoreGlobal1DF32Indirect`: 1D indirect (gather/scatter) using stack-provided indices.
- Stack protocol: push values (store only), then push indices, then call the indirect instruction.
- Indices are f32-bitcast-to-i32 on the stack; `advance_ptr_cond_state` bitcasts back to i32 for addressing.
- **Width-lane iteration contract**: The base `emit` loop in `_LoadStoreGlobalBase` calls `advance_ptr_cond_state(w + 1, ...)` after processing lane `w`, because `advance` receives the lane index it is *setting up* (not the one just completed). `init_ptr_cond_state` bootstraps lane 0 by calling `advance(w=0, ...)`.

## Indirect Dispatch Mode

- The kernel accepts a `flag: i32` as its 5th argument.
- **Direct mode** (`flag <= 0`): All warps share the same `arg(0)` (bytecode pointer) and `arg(1)` (TensorInfo pointer) — the original behavior.
- **Indirect mode** (`flag > 0`): `arg(0)` and `arg(1)` are treated as **pointer tables** indexed by `logical_program_idx()`, so each warp loads its own program and tensor info via `ptr = ptrs[idx]`.
- Resolution uses `if_else` + phi to avoid speculative loads on invalid pointers in direct mode.
- The early-exit bounds check (`logical_program_idx() >= arg(3)`) applies in both modes — in indirect mode, `arg(3)` is the total number of programs in the table.
- After pointer resolution, the rest of the interpreter (dispatch loop, tensor info loading, instruction execution) is unchanged.

## Bytecode Instruction Format

- Each instruction is a 2-word pair `[opcode: i32, operand: i32]` in the bytecode stream.
- PC always advances by 2 after each instruction.
- Instructions with no meaningful operand use `operand = 0`.
- Opcodes defined in `INSNS` dict in `gint/kernel/interpreter/main.py`.

## Kernel Signature

```
void geval(i32* code, TensorInfo* tinfo, i32 num_tensors, i32 grid_dim, i32 flag)
```
- `flag <= 0`: direct mode — `code` and `tinfo` are used as-is by all warps.
- `flag > 0`: indirect mode — `code` and `tinfo` are cast to `i32**` / `TensorInfo**` and indexed per-warp.

## Adding New Instructions

1. Implement the instruction class in `gint/kernel/interpreter/instructions/` (subclass `DefaultControlInstruction` or `DefaultControlOperandInstruction`).
2. Add to `INSNS` dict in `gint/kernel/interpreter/main.py` with the next available opcode.
3. Add a frontend function in `gint/host/frontend.py` decorated with `@_bc`.
4. Run `./generate.sh` (NVIDIA) and/or `./generate_amdgcn.sh` (AMD) to regenerate binaries.
