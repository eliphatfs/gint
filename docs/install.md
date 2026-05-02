# Installation & Build

## Dependencies

Runtime dependencies (declared in `pyproject.toml`):
- `numpy`, `llvmlite>=0.43`, `cuda-bindings>=12.6`, `rich`
- Optional: `torch>=2.0` (for conductor / `torch.compile` backend)
- Optional: `hip-python>=6.0` (for AMD HIP backend)

For cuda-bindings version compatibility details, see `docs/platforms.md`.

## Install

Built with `hatchling`. Requires Python >=3.10.

```bash
pip install -e .          # core (needs cuda-bindings, numpy, llvmlite, rich)
pip install -e ".[torch]" # with torch.compile backend

# AMD HIP backend (hip-python is only on Test PyPI due to AMD's packaging policy)
python3 -m pip install -i https://test.pypi.org/simple "hip-python>=6.0.0"
```

> **Note**: `hip-python` is not declared as a pip dependency because AMD only publishes real packages to Test PyPI (the PyPI entry is a dummy). This needs to be documented for end users when gint is published.

Entry points:
- `gint-gen-llir`: LLVM IR generation script.
- `gint-driver`: driver utility.

## Code Generation

```bash
# NVIDIA: Generate LLVM IR and compile to PTX/fatbin for all compute capabilities
# Requires: CUDA Toolkit 12.8+, LLVM/clang 20, llvmlite, rich, numpy
./generate.sh

# Generates files:
# - artifact/gint.ptx
# - artifact/gint.fatbin
# - artifact/gint.fatbin.xz (compressed, used by runtime)
# - gint/host/cuda/gint.fatbin.xz (deployed version, NOT tracked in git)

# AMD: Generate LLVM IR and compile to HSACO for RDNA3+ targets
# Requires: ROCm 7.0+, OCML/OCKL bitcode
./generate_amdgcn.sh

# Generates files:
# - artifact/gint_gfx*.s (per-target GCN assembly text)
# - artifact/gint_gfx*.o (per-target code objects)
# - artifact/gint_amdgcn.fatbin (bundled fat binary, all targets)
# - gint/host/hip/gint.fatbin.xz (deployed version, NOT tracked in git)

# Direct LLVM IR generation (for debugging/inspection)
gint-gen-llir -t llir      # NVPTX LLVM IR (default)
gint-gen-llir -t amdgcn --gfx gfx1100  # AMDGCN assembly
```

## CI / Wheel Build

GitHub Actions workflow (`.github/workflows/build.yml`) — **manual dispatch only** (`workflow_dispatch`):
1. `generate-fatbin` job: runs in `nvidia/cuda:12.8.1-devel-ubuntu22.04` container, installs LLVM 18, generates the fatbin.
2. `build-wheel` job: downloads fatbin artifact, builds a `py3-none-any` wheel with hatchling, verifies fatbin is included.

The wheel is pure Python (no native extensions) so only one Python version is needed for the build. The fatbin is included via hatchling's `artifacts` config which overrides `.gitignore` exclusion.
