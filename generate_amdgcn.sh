set -xe
mkdir -p artifact

TARGETS="gfx1100 gfx1101 gfx1200 gfx1201"

# Generate GCN assembly text per target
for gfx in $TARGETS; do
    gint-gen-llir -t amdgcn --gfx $gfx -o artifact/gint_${gfx}.s
done

# Compile each .s to relocatable object
for gfx in $TARGETS; do
    clang -c --target=amdgcn-amd-amdhsa -mcpu=$gfx \
        -x assembler artifact/gint_${gfx}.s \
        -o artifact/gint_${gfx}.o
done

# Link each .o into a shared object (hipModuleLoadData needs linked ELF, not relocatable .o)
for gfx in $TARGETS; do
    ld.lld -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -plugin-opt=-amdgpu-internalize-symbols \
        --lto-partitions=8 -plugin-opt=mcpu=$gfx -plugin-opt=O3 --lto-CGO3 \
        --whole-archive -o artifact/gint_${gfx}.out artifact/gint_${gfx}.o --no-whole-archive
done

# Bundle all .out into fat binary via clang-offload-bundler
clang-offload-bundler -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1100,hipv4-amdgcn-amd-amdhsa--gfx1101,hipv4-amdgcn-amd-amdhsa--gfx1200,hipv4-amdgcn-amd-amdhsa--gfx1201 \
    -input=/dev/null \
    -input=artifact/gint_gfx1100.out \
    -input=artifact/gint_gfx1101.out \
    -input=artifact/gint_gfx1200.out \
    -input=artifact/gint_gfx1201.out \
    -output=artifact/gint_amdgcn.fatbin

# Compress and deploy
xz -efk artifact/gint_amdgcn.fatbin
mkdir -p gint/host/hip
cp -v artifact/gint_amdgcn.fatbin.xz gint/host/hip/gint.fatbin.xz
