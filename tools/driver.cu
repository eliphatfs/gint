#include <cuda.h>
#include <stdio.h>


#define CHECK_CUDA_DRIVER(call)                                               \
do {                                                                          \
    CUresult result = call;                                                   \
    if (result != CUDA_SUCCESS) {                                             \
        const char* errorStr = NULL;                                          \
        const char* errorName = NULL;                                         \
        cuGetErrorName(result, &errorName);                                   \
        cuGetErrorString(result, &errorStr);                                  \
        fprintf(stderr, "CUDA Driver API Error at %s:%d - %s (%s)\n",         \
                __FILE__, __LINE__,                                           \
                errorName ? errorName : "Unknown",                            \
                errorStr ? errorStr : "Unknown CUDA error");                  \
        exit(EXIT_FAILURE); /* Or handle error appropriately */               \
    }                                                                         \
} while(0)


void * read_ptx(const char* filename)
{
    void * buffer = 0;
    long length;
    FILE * f = fopen (filename, "rb");

    if (f)
    {
        fseek (f, 0, SEEK_END);
        length = ftell (f);
        fseek (f, 0, SEEK_SET);
        buffer = calloc (length + 1, sizeof(char));
        if (buffer)
        {
            fread (buffer, 1, length, f);
        }
        fclose (f);
    }
    return buffer;
}


int main()
{
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
    void * ptx = read_ptx("test.ptx");

    // JIT compile a null-terminated PTX string
    CHECK_CUDA_DRIVER(cuModuleLoadData(&module, ptx));

    // Get a handle to the "myfunction" kernel function
    CHECK_CUDA_DRIVER(cuModuleGetFunction(&function, module, "geval"));

    int x = 0;

    void* kernelParams[1] = {&x};

    CHECK_CUDA_DRIVER(cuLaunchKernel(
        function,
        1, 1, 1,
        1, 1, 1,
        0,
        (CUstream)0,
        kernelParams,
        NULL
    ));
    CHECK_CUDA_DRIVER(cuStreamSynchronize((CUstream)0));

    free(ptx);
    
    CHECK_CUDA_DRIVER(cuModuleUnload(module));
    CHECK_CUDA_DRIVER(cuCtxDestroy(context));
    return 0;
}
