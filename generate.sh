mkdir -p artifact
gint-gen-llir -t ptx -c &&
gint-gen-llir -t ptx --cc 70 -o artifact/gint.ptx

nvcc -lineinfo -fatbin --ptxas-options=-v -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 artifact/gint.ptx -o artifact/gint.fatbin

xz -efk artifact/gint.fatbin 
cp -v artifact/gint.fatbin.xz gint/host/cuda/gint.fatbin.xz
