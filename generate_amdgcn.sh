set -xe
mkdir -p artifact

# Generate per-target HSACO code objects for RDNA3 discrete GPUs
for gfx in gfx1100 gfx1101 gfx1102; do
    gint-gen-llir -t amdgcn --gfx $gfx -o artifact/gint_${gfx}.hsaco
done

# Generic targets for forward compatibility:
#   gfx11-generic covers all RDNA3 + RDNA3.5 (gfx1100-gfx1153)
#   gfx12-generic covers all RDNA4 (gfx1200, gfx1201)
gint-gen-llir -t amdgcn --gfx gfx11-generic -o artifact/gint_gfx11-generic.hsaco
gint-gen-llir -t amdgcn --gfx gfx12-generic -o artifact/gint_gfx12-generic.hsaco

# Bundle into a single HIP fat binary (hipfb) using clang-offload-bundler
clang-offload-bundler --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1100,hipv4-amdgcn-amd-amdhsa--gfx1101,hipv4-amdgcn-amd-amdhsa--gfx1102,hipv4-amdgcn-amd-amdhsa--gfx11-generic,hipv4-amdgcn-amd-amdhsa--gfx12-generic \
  --input=artifact/gint_gfx1100.hsaco \
  --input=artifact/gint_gfx1101.hsaco \
  --input=artifact/gint_gfx1102.hsaco \
  --input=artifact/gint_gfx11-generic.hsaco \
  --input=artifact/gint_gfx12-generic.hsaco \
  --output=artifact/gint.hipfb

xz -efk artifact/gint.hipfb
mkdir -p gint/host/hip
cp -v artifact/gint.hipfb.xz gint/host/hip/gint.hipfb.xz
