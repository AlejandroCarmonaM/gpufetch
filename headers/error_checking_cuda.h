#ifndef ERROR_CHECKING_CUDA_H
#define ERROR_CHECKING_CUDA_H

#include <cuda_runtime.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

// CUDA ERROR CHECKING
template <typename T>
void check_cuda(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err) << "(" << cudaGetErrorString(err) << ") \"" << func << "\" \n";
        exit(1);
    }
}

// Get GPU name
void get_gpu_name(char *gpu_name) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    strcpy(gpu_name, deviceProp.name);
}

#endif // ERROR_CHECKING_CUDA_H