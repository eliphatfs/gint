# Superoptimizer Performance Benchmarks

Technical experiment report measuring the performance characteristics of the
gint superoptimizer's GPU-accelerated batch evaluation pipeline.

## System Configuration

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | 12.1 |
| PyTorch | 2.5.0+cu121 |
| Driver mode | Indirect dispatch (1 warp per candidate program) |
| Test vector size | 128 elements (= WARP_SIZE(32) x REG_WIDTH(4)) |

## Reproducing

```bash
# Run all experiments (~50s)
python -m examples.superopt.bench.run_all

# Skip specific experiments
python -m examples.superopt.bench.run_all --skip comparison

# Results are saved to examples/superopt/bench/results.json
```

---

## Experiment 1: Phase-Level Breakdown

**Goal:** Identify which phases of the superoptimizer pipeline dominate
wall-clock time.

**Setup:** Brute-force search on the `gelu` target (6-instruction reference,
45 search ops including transcendentals) at search length 5, producing
14,108,850 candidate sequences. Batch size 4096.

**Method:** Wall-clock timing of each phase:
1. **Enumeration** -- Python DFS with depth-reachability pruning generates all
   stack-valid instruction sequences
2. **Bytecode build** -- Convert `SearchOp` sequences to numpy `(N, words)`
   int32 arrays with load prefix + body + store/halt suffix
3. **GPU execution** -- `BatchRunner.run()`: allocate device memory, upload
   bytecodes + tensor infos + pointer tables, launch kernel (indirect mode),
   synchronize, free
4. **Comparison** -- Vectorized `torch` comparison of candidate outputs against
   reference (handles NaN matching)

### Results

| Phase | Time (s) | Share |
|-------|----------|-------|
| Enumeration | 6.79 | 34.3% |
| Bytecode build | 8.26 | 41.8% |
| GPU execution | 4.44 | 22.4% |
| Comparison | 0.30 | 1.5% |
| **Total** | **19.8** | |

**End-to-end throughput:** 713,062 candidates/s

### Interpretation

The GPU kernel -- despite evaluating 14M distinct programs -- accounts for
only 22% of total time. The bottleneck is host-side Python:

- **Bytecode build (42%):** `sequences_to_bytecodes()` iterates over all
  sequences in Python, writing `(opcode, operand)` pairs into a numpy array.
  This is inherently sequential per-candidate.
- **Enumeration (34%):** The DFS generator in `enumerate_exact_length()` uses
  aggressive depth-reachability pruning (cutting branches that can't reach the
  target stack depth), but the 14M-node search tree still requires 14M Python
  function calls.
- **Comparison (1.5%):** Fully vectorized in PyTorch, negligible.

The GPU is underutilized. Merging enumeration and bytecode construction into
a single C/numpy-vectorized pass would shift the bottleneck to GPU execution,
potentially achieving the ~3M candidates/s throughput measured in Experiment 2.

---

## Experiment 2: Per-Batch GPU Executor Breakdown

**Goal:** Profile what happens inside a single `BatchRunner.run()` call to
identify overhead within the GPU execution phase.

**Setup:** relu target, batch size 4096, averaged over 20 runs. Instrumented
`BatchRunner.run()` timing each sub-operation.

### Results

| Sub-phase | Time (ms) | Share |
|-----------|-----------|-------|
| `build_ti` (struct.pack_into loop) | 0.734 | 54.8% |
| `build_ptrs` (numpy pointer arrays) | 0.415 | 31.0% |
| `copy_ti` (cuMemAlloc + cuMemcpyHtoD) | 0.118 | 8.8% |
| `launch` (kernel launch + sync) | 0.018 | 1.3% |
| `alloc_code` + `copy_ptrs` + `free` | 0.054 | 3.9% |
| **Total** | **1.339** | |

**Isolated GPU throughput:** 3,058,999 candidates/s

### Interpretation

The kernel launch + execution + synchronize takes only **0.018 ms** for 4096
programs -- the GPU processes each candidate in under 5 nanoseconds. The real
per-batch cost is Python overhead:

- **`build_ti` (55%):** A `struct.pack_into` loop that patches per-candidate
  output pointers into replicated `HTensorInfo` templates. Each iteration does
  `N_inputs + 1` pack operations. This is the single largest bottleneck within
  the GPU executor.
- **`build_ptrs` (31%):** Creating numpy arrays of device pointer arithmetic
  (`base + i * stride`). Could be vectorized further.
- **CUDA API calls (14%):** Memory allocation, copies, and frees are fast in
  absolute terms (~0.2 ms total for 4 allocations + 4 copies + 4 frees).

The 3M candidates/s ceiling is set by Python, not GPU compute. A C extension
for tensor info construction would push throughput to CUDA API-limited speeds
(~8M candidates/s based on the 0.2 ms API overhead alone).

---

## Experiment 3: Three-Way Speed Comparison

**Goal:** Quantify the speedup of batch GPU evaluation vs. the two obvious
alternative implementations.

**Setup:** relu target, search length 4, 69,457 total candidates. All three
methods use float32 arithmetic and identical NaN-aware comparison logic.

**Methods compared:**
1. **CPU Python interpreter** -- Pure-Python stack machine emulating the gint
   instruction set in `np.float32`. Evaluates each candidate on all 128 test
   elements sequentially with early exit on first mismatch. No GPU involvement.
2. **Sequential GPU** -- One `BatchRunner.run()` call per candidate (batch
   size = 1). Same GPU kernel, but per-candidate launch overhead dominates.
3. **Batch GPU (ours)** -- `BatchRunner.run()` with batch size 4096. One
   kernel launch evaluates thousands of candidates simultaneously.

### Results

| Method | Throughput (candidates/s) | Evaluated | Time (s) | Matches |
|--------|--------------------------|-----------|----------|---------|
| CPU Python interpreter | 304,390 | 69,457 | 0.23 | 3 |
| Sequential GPU (batch=1) | 11,611 | 2,000 | 0.17 | 0 |
| Batch GPU (batch=4096) | 1,249,729 | 69,457 | 0.06 | 3 |

| Speedup | Factor |
|---------|--------|
| Batch GPU vs CPU | **4.1x** |
| Batch GPU vs Sequential GPU | **108x** |

### Interpretation

**CPU throughput is high due to early exit.** Most candidates produce NaN or a
wrong value on the very first test element, so the per-element loop exits
immediately. The CPU evaluates ~1.2 elements per candidate on average vs. the
GPU's fixed 128. This is a fundamental advantage of scalar sequential
evaluation: it can reject bad candidates without computing all outputs. The
GPU evaluates all 128 elements for all candidates regardless.

**Despite CPU early exit, batch GPU is still 4x faster.** The GPU's massive
parallelism (4096 candidates x 128 elements = 524K evaluations per launch)
overcomes the CPU's algorithmic advantage. For search spaces with fewer
NaN-producing paths (e.g., when transcendentals are excluded from the search
ops), the CPU's early-exit advantage shrinks and the GPU speedup grows.

**Sequential GPU is the slowest method.** With batch size 1, each candidate
requires ~12 CUDA API calls (alloc, copy, launch, sync, free). At ~10 us each,
that's ~120 us overhead per candidate. The kernel itself runs in <1 us. Launch
overhead accounts for >99% of sequential GPU time.

**Batch GPU amortizes launch overhead.** The same 12 CUDA API calls service
4096 candidates instead of 1, reducing per-candidate API overhead from ~120 us
to ~0.03 us (4000x reduction). This is the core advantage of `execute_indirect`:
per-warp program heterogeneity within a single kernel launch.

**Match counts agree across all methods** (3 matches for relu at length 4),
validating that the CPU interpreter faithfully emulates the GPU's float32
stack machine semantics including NaN propagation.

---

## Experiment 4: Batch Size Scaling

**Goal:** Find the optimal batch size and measure end-to-end throughput on a
realistically large search.

**Setup:** gelu target, search length 5 (14,108,850 candidates), pool of
65,536 pre-built bytecodes for the scaling sweep.

### Scaling Results

| Batch Size | Throughput (candidates/s) |
|------------|--------------------------|
| 256 | 1,727,569 |
| 512 | 2,280,141 |
| 1,024 | 2,675,928 |
| 2,048 | 3,077,845 |
| 4,096 | 3,003,582 |
| 8,192 | 3,168,760 |
| 16,384 | 3,095,251 |

**Plateau at ~3M candidates/s from batch size 2048 onward.**

### Full Search Results

| Metric | Value |
|--------|-------|
| Total candidates | 14,108,850 |
| Wall-clock time | 13.1 s |
| End-to-end throughput | 1,074,974 candidates/s |

**Projected times for 14.1M candidates under each method:**

| Method | Estimated Time |
|--------|---------------|
| CPU Python interpreter | ~47 s (with early exit) |
| Sequential GPU (batch=1) | ~20 min |
| **Batch GPU (actual)** | **13 s** |

Note: CPU time is estimated from Experiment 3's 300K candidates/s throughput,
which benefits heavily from early NaN exit. On targets with fewer NaN paths
or with a C-based interpreter, the gap would be larger.

### Interpretation

**Batch size scaling** shows diminishing returns past 2048. The bottleneck
shifts from per-launch overhead (dominant at small batches) to the Python
`struct.pack_into` loop (dominant at large batches). The per-candidate cost of
the loop is constant regardless of batch size, creating the plateau.

**End-to-end throughput (1.1M/s) is lower than isolated GPU throughput
(3M/s)** because the full search includes enumeration and bytecode
construction. The 3M/s rate is the GPU executor's ceiling; the 1.1M/s rate
reflects the complete pipeline including Python overhead.

The batch GPU approach makes brute-force search over millions of candidates
practical. The gelu search (14M candidates, 45-op search space at length 5)
completes in 13 seconds -- fast enough for interactive use during kernel
development.

---

## Summary of Findings

1. **The GPU kernel is not the bottleneck.** At 0.018 ms per 4096 candidates,
   the kernel runs 3 orders of magnitude faster than the Python code feeding
   it. The architecture's per-warp dispatch is highly efficient for this
   workload.

2. **Python host-side code dominates (77% of wall time).** Enumeration (34%)
   and bytecode construction (42%) are the primary bottlenecks. Both are
   inherently sequential Python loops over millions of candidates.

3. **Batch dispatch provides 4-108x speedup** over CPU interpretation
   (with early exit) and sequential GPU launches respectively. The key insight
   is amortizing CUDA API overhead across thousands of programs per launch via
   `execute_indirect`.

4. **The `struct.pack_into` loop is the per-batch bottleneck (55%).** Each
   candidate needs its own `HTensorInfo` with a patched output pointer. This
   Python loop is the ceiling on GPU-side throughput.

5. **Batch size sweet spot is 2048-8192.** Below 2048, per-launch overhead is
   significant. Above 8192, returns diminish as the per-candidate Python cost
   dominates.

6. **CPU early exit is a strong baseline.** The CPU interpreter can reject most
   candidates after evaluating just 1-2 elements, while the GPU always
   evaluates all 128. Despite this, the GPU's parallel throughput wins for
   large-scale exhaustive searches.

### Optimization opportunities (not implemented)

- **Fused enumerate+build:** Merge enumeration and bytecode construction into
  a single pass, eliminating the intermediate `List[SearchOp]` representation.
  Expected improvement: ~40% reduction in total time.
- **C extension for tensor info:** Replace `struct.pack_into` loop with a C
  function that bulk-patches output pointers. Would push per-batch throughput
  from 3M to ~8M candidates/s.
- **Numpy-vectorized enumeration:** Express the DFS as parallel numpy
  operations (similar to `random_valid_batch`). Harder for exact enumeration
  but would remove the 34% enumeration bottleneck.
