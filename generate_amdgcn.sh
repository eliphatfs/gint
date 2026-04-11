set -xe
mkdir -p artifact

# Generate per-target HSACO code objects
for gfx in gfx1100 gfx1101 gfx1102 gfx11-generic gfx12-generic; do
    gint-gen-llir -t amdgcn --gfx $gfx -o artifact/gint_${gfx}.hsaco
    xz -efk artifact/gint_${gfx}.hsaco
done

# Pack all compressed HSACOs into a single zip archive
python3 -c "
import zipfile, glob
with zipfile.ZipFile('artifact/gint_amdgcn.zip', 'w', zipfile.ZIP_STORED) as zf:
    for f in sorted(glob.glob('artifact/gint_*.hsaco.xz')):
        zf.write(f, f.split('/')[-1])
"

mkdir -p gint/host/hip
cp -v artifact/gint_amdgcn.zip gint/host/hip/gint_amdgcn.zip
