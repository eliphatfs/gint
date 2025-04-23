#include <cuda.h>
#include <stdio.h>

#define CHECK_CUDA_DRIVER(call)                                                \
  do {                                                                         \
    CUresult result = call;                                                    \
    if (result != CUDA_SUCCESS) {                                              \
      const char *errorStr = NULL;                                             \
      const char *errorName = NULL;                                            \
      cuGetErrorName(result, &errorName);                                      \
      cuGetErrorString(result, &errorStr);                                     \
      fprintf(stderr, "CUDA Driver API Error at %s:%d - %s (%s)\n", __FILE__,  \
              __LINE__, errorName ? errorName : "Unknown",                     \
              errorStr ? errorStr : "Unknown CUDA error");                     \
      exit(EXIT_FAILURE); /* Or handle error appropriately */                  \
    }                                                                          \
  } while (0)

void *read_ptx(const char *filename) {
  void *buffer = 0;
  long length;
  FILE *f = fopen(filename, "rb");

  if (f) {
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = calloc(length + 1, sizeof(char));
    if (buffer) {
      fread(buffer, 1, length, f);
    }
    fclose(f);
  }
  return buffer;
}

void ptx_link(CUmodule *phModule, CUfunction *phKernel, CUlinkState *lState,
              void *ptx_source, const char *fn_name) {
  CUjit_option options[6];
  void *optionVals[6];
  float walltime;
  char error_log[8192], info_log[8192];
  unsigned int logSize = 8192;
  void *cuOut;
  size_t outSize;
  int myErr = 0;

  // Setup linker options
  // Return walltime from JIT compilation
  options[0] = CU_JIT_WALL_TIME;
  optionVals[0] = (void *)&walltime;
  // Pass a buffer for info messages
  options[1] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[1] = (void *)info_log;
  // Pass the size of the info buffer
  options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[2] = (void *)(long)logSize;
  // Pass a buffer for error message
  options[3] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[3] = (void *)error_log;
  // Pass the size of the error buffer
  options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[4] = (void *)(long)logSize;
  // Make the linker verbose
  options[5] = CU_JIT_LOG_VERBOSE;
  optionVals[5] = (void *)1;

  // Create a pending linker invocation
  CHECK_CUDA_DRIVER(cuLinkCreate(6, options, optionVals, lState));

  // Load the PTX from the ptx file
  myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)ptx_source,
                        strlen((const char *)ptx_source) + 1, 0, 0, 0, 0);

  if (myErr != CUDA_SUCCESS) {
    // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option
    // above.
    fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
  }

  // Complete the linker step
  CHECK_CUDA_DRIVER(cuLinkComplete(*lState, &cuOut, &outSize));

  // Linker walltime and info_log were requested in options above.
  // printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime,
  // info_log);

  // Load resulting cuBin into module
  CHECK_CUDA_DRIVER(cuModuleLoadData(phModule, cuOut));

  // Locate the kernel entry poin
  CHECK_CUDA_DRIVER(cuModuleGetFunction(phKernel, *phModule, fn_name));

  // Destroy the linker invocation
  CHECK_CUDA_DRIVER(cuLinkDestroy(*lState));
}

int main() {
  CUdevice device;
  CUcontext context;

  // Initialize the driver API
  CHECK_CUDA_DRIVER(cuInit(0));
  // Get a handle to the first compute device
  CHECK_CUDA_DRIVER(cuDeviceGet(&device, 0));
  // Create a compute device context
  CHECK_CUDA_DRIVER(cuCtxCreate(&context, 0, device));

  CUmodule module;
  CUfunction function;
  CUlinkState lState;
  void *ptx = read_ptx("test.ptx");

  ptx_link(&module, &function, &lState, ptx, "geval");

  int x = 0;

  void *kernelParams[1] = {&x};

  CHECK_CUDA_DRIVER(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, (CUstream)0,
                                   kernelParams, NULL));
  CHECK_CUDA_DRIVER(cuStreamSynchronize((CUstream)0));

  free(ptx);

  CHECK_CUDA_DRIVER(cuModuleUnload(module));
  CHECK_CUDA_DRIVER(cuCtxDestroy(context));
  return 0;
}
