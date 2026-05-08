set -xe
REG_WIDTH=${REG_WIDTH:-4}
export GINT_REG_WIDTH=$REG_WIDTH
mkdir -p artifact
gint-gen-llir -t ptx -c --reg-width "$REG_WIDTH" >/dev/null || gint-gen-llir -t ptx -c --reg-width "$REG_WIDTH"
gint-gen-llir -t ptx --cc 70 --reg-width "$REG_WIDTH" -o artifact/gint.ptx

nvcc -lineinfo -fatbin --ptxas-options=-v \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  artifact/gint.ptx -o artifact/gint.fatbin

xz -efk artifact/gint.fatbin
cp -v artifact/gint.fatbin.xz gint/host/cuda/gint.fatbin.xz
