#include <stdio.h>
#include "./headers/cache_benchmark.h"

#define TC_SM_TURING 8 
#define TC_SM_AMPERE 4
#define TC_SM_ADA_LOVELACE 4

#define CUDA_CORES_PER_SM_TURING 64
#define CUDA_CORES_PER_SM_AMPERE 128
#define CUDA_CORES_PER_SM_ADA_LOVELACE 128

/*IDEAS: To map certain static info not available at runtime.
Example: Tensor Core Number*/

// Helper function to convert compute capability to core count
// This is generally provided by NVIDIA, and values need to be updated based on the latest GPUs
int _ConvertSMVer2Cores(int major, int minor) {
    // This table needs to be updated for newer compute capabilities
    // Values for existing compute capabilities can be found in CUDA documentation or NVIDIA forums
    if (major == 7 && minor == 5) { // For Turing architecture
        return CUDA_CORES_PER_SM_TURING; 
    }
    if (major == 8 && minor == 6) { // For Ampere architecture
        return CUDA_CORES_PER_SM_AMPERE; 
    }
    if (major == 8 && minor == 9) { // For Marykeller architecture
        return CUDA_CORES_PER_SM_ADA_LOVELACE;
    }
    // Add more cases as needed
    // If unknown, return a default value
    return -1; // Indicates unknown number of cores
}


// Helper function to get number of tensor cores
void get_tensor_cores(cudaDeviceProp deviceProp){
    int tensor_cores_per_sm = 0;
    if (deviceProp.major == 7 && deviceProp.minor == 5) { // For Turing architecture
        tensor_cores_per_sm = TC_SM_TURING;
        //Tensor Core Architecture Generation
        printf("  Tensor Core Architecture: Turing\n");
        printf("  Number of Tensor Cores per SM: %d\n", tensor_cores_per_sm);
        printf("  Total number of Tensor Cores: %d\n", tensor_cores_per_sm * deviceProp.multiProcessorCount);
    }
    if (deviceProp.major == 8 && deviceProp.minor == 6) { // For Ampere architecture
        tensor_cores_per_sm = TC_SM_AMPERE;
        //Tensor Core Architecture Generation
        printf("  Tensor Core Architecture: Ampere\n");
        printf("  Number of Tensor Cores per SM: %d\n", tensor_cores_per_sm);
        printf("  Total number of Tensor Cores: %d\n", tensor_cores_per_sm * deviceProp.multiProcessorCount);
    }
    if (deviceProp.major == 8 && deviceProp.minor == 9) { // For Ada Lovelace architecture
        tensor_cores_per_sm = TC_SM_ADA_LOVELACE;
        //Tensor Core Architecture Generation
        printf("  Tensor Core Architecture: Ada Lovelace\n");
        printf("  Number of Tensor Cores per SM: %d\n", tensor_cores_per_sm);
        printf("  Total number of Tensor Cores: %d\n", tensor_cores_per_sm * deviceProp.multiProcessorCount);
    }

    if(tensor_cores_per_sm == 0){
        printf("  Number of Tensor Cores per SM: Unknown\n");
        printf("  Total number of Tensor Cores: Unknown\n");
    }
}

// Function to print device properties in a legible way
void printDeviceProperties(cudaDeviceProp deviceProp, int device) {
        printf("Device Number: %d\n", device);
    printf("  Device name: %s\n", deviceProp.name);
    printf("\nTECHNICAL SPECIFICATIONS\n");
    printf("\n");
    //clock rate in GHz (clockRate returns KHz)
    printf("  Clock rate (GHz): %.2f\n", deviceProp.clockRate / 1.0e6);
    //memory bus width
    printf("  Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
    //memory bandwidth
    printf("  Peak Memory Bandwidth (GB/s): %.2f\n\n",
           2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
    //global memory
    printf("  Total global memory (GB): %.2f\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    //cache size
    printf("  L2 Cache Size (KB): %d\n", deviceProp.l2CacheSize / 1024);
    //shared memory
    deviceProp.localL1CacheSupported ? printf("  L1 cache supported\n") : printf("  L1 cache not supported\n");
    // For compute capability >= 2.x, discuss shared memory and L1 cache configuration
    if (deviceProp.major >= 2) {
       printf(" Note: L1 cache and shared memory are configured together. Use cudaFuncSetCacheConfig or cudaDeviceSetCacheConfig to adjust.\n");
    }
    printf("  Shared memory per block (KB): %zu\n", deviceProp.sharedMemPerBlock / 1024);
    //registers
    printf("  Registers per block: %d\n", deviceProp.regsPerBlock);
    //warp size
    printf("  Warp size: %d\n", deviceProp.warpSize);
    //max threads per multiprocessor
    printf("  Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    //Check Tensor Cores
    deviceProp.major >= 7 ? printf("  Tensor Cores supported\n") : printf("  Tensor Cores not supported\n");
    //Number of Tensor cores
    if(deviceProp.major >= 7){
        get_tensor_cores(deviceProp);
    }
    printf("  Total number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("  Total number of cores: %d\n", deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max threads dimensions (x, y, z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Max grid size (x, y, z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("\n");
    //TO BE TESTED: peak FLOPS
    printf("Peak Performance CUDA Cores (No Tensor Cores)\n");
    printf("  Peak TFLOPS FP16: %f\n", 4.0 * (double)_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount * deviceProp.clockRate / 1.0e9);
    printf("  Peak TFLOPS FP32: %f\n", 2.0 * (double)_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount * deviceProp.clockRate / 1.0e9);
    printf("  Peak TFLOPS FP64: %f\n", (double)_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount * deviceProp.clockRate / 1.0e9);

    printf("\nPERFORMANCE SPECIFICATIONS (benchmarked)\n");
    double L2_bw = run_cache_benchmark(deviceProp.l2CacheSize);
    printf("  L2 Cache Bandwidth (GiB/s): %.2f\n", L2_bw); //por que si le sumo solo 6528 en vez de 6528*1024 va rapidisima???
    // GLOBAL MEMORY BANDWIDTH
    size_t bench_mem = deviceProp.totalGlobalMem*0.1;
    double global_bw = run_cache_benchmark(bench_mem);
    printf("  Global Memory Bandwidth (GiB/s): %.2f\n", global_bw);


    printf("\n");
}

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printDeviceProperties(prop, i);
    }

    return 0;
}
