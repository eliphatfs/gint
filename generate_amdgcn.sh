set -xe
mkdir -p artifact

# Generate per-target HSACO code objects
for gfx in gfx1100 gfx1101 gfx1102 gfx11-generic gfx12-generic; do
    gint-gen-llir -t amdgcn --gfx $gfx -o artifact/gint_${gfx}.hsaco
    xz -efk artifact/gint_${gfx}.hsaco
done

# Pack all compressed HSACOs into a single zip archive
cd artifact
zip -0 gint_amdgcn.zip gint_*.hsaco.xz
cd ..

mkdir -p gint/host/hip
cp -v artifact/gint_amdgcn.zip gint/host/hip/gint_amdgcn.zip
