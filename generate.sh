set -xe
mkdir -p artifact
DISPATCH_MODE=${DISPATCH_MODE:-switch}
gint-gen-llir -t ptx -c --dispatch-mode "$DISPATCH_MODE" >/dev/null || gint-gen-llir -t ptx -c --dispatch-mode "$DISPATCH_MODE"
gint-gen-llir -t ptx --cc 70 --dispatch-mode "$DISPATCH_MODE" -o artifact/gint.ptx

nvcc -lineinfo -fatbin --ptxas-options=-v \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  artifact/gint.ptx -o artifact/gint.fatbin

xz -efk artifact/gint.fatbin
cp -v artifact/gint.fatbin.xz gint/host/cuda/gint.fatbin.xz
