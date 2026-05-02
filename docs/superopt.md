# Superoptimizer

A GPU-accelerated superoptimizer that finds shorter equivalent bytecode sequences for gint instruction patterns. Uses `execute_indirect` to evaluate thousands of candidate programs per kernel launch — no kernel changes needed.

Source: `examples/superopt/`.

## Usage

```bash
# Search for shorter equivalent bytecode sequences for a target
python -m examples.superopt relu

# Run on all targets
python -m examples.superopt --all

# List available targets
python -m examples.superopt --list
```

## Architecture

- **`opcodes.py`**: search space definition — 33+ ops with stack effects (min_depth, net_effect). Expandable with transcendentals.
- **`candidates.py`**: DFS enumeration with depth-reachability pruning. For length 5 unary targets, prunes 33^5=40M down to ~1.5M valid sequences. Also has numpy-vectorized random batch generation for stochastic search.
- **`executor.py`**: `BatchRunner` — concatenates all candidate bytecodes into one device allocation, builds per-candidate tensor infos with only the output pointer patched, uses a single indirect-mode kernel launch. ~8 CUDA API calls regardless of batch size.
- **`search.py`**: brute-force (exhaustive, length 1..N) and stochastic (random mutation hill climbing) modes. Candidates verified with 10 additional random test vector sets.
- **`targets.py`**: reference sequences from `op_registry.py` to optimize.

## Key insight discovered

`fselect` treats `peek(0)` as a float condition (`>0` → true branch). This means explicit comparison-against-zero patterns (`fpush(0) → flt/fgt → fselect`) can often be eliminated:
- **relu**: `x * float(x > 0)` via `dup → fpush(0) → flt → fmul` (4 insns, was 5).
- **abs**: `select(-x > 0, -x, x)` via `dup → fneg → dup → fselect` (4 insns, was 7).
- **leaky_relu**: `dup → fmulimm(ns) → dupx1 → fselect` exploits that `ns*x > 0 ↔ x > 0` for positive slopes (4 insns, was 7).
- **gelu, silu**: already optimal (6 and 5 insns respectively).
