// cache_benchmark.h
#ifndef CACHE_BENCHMARK_H
#define CACHE_BENCHMARK_H

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_timekeeper.h"
#include "error_checking_cuda.h"

#define WARMUP_ITERATIONS 10
#define ITERATIONS 50
#define FRACTION_BIDIRECTIONAL 0.27
#define FRACTION_COPY 0.4



// KERNELS

// Copy benchmark
__global__ void cacheBandwidthCopyBenchmark(float *a, float *b, int N) {
    int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = gridDim.x * blockDim.x;

    //vector copy
    for (int i = indexWithinTheGrid; i < N; i += gridStride) {
        b[i] = a[i];
    }
    
}

__global__ void bidirectionalBandwidthBenchmark(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride) {
        c[i] = a[i] + b[i]; // Example operation that reads from a and b, then writes to c
    }
}


double run_bidirectionalBandwidthBenchmark(size_t cache_size) {
    int elements = (cache_size / sizeof(float)) * FRACTION_BIDIRECTIONAL; // 3 arrays each of size cache_size/FRACTION
    float *a, *b, *c;

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    // Get the number of SMs on the current GPU
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&a, elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&b, elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&c, elements * sizeof(float)));

    // Use cudaMemset to initialize device memory
    checkCudaErrors(cudaMemset(a, 1, elements * sizeof(float)));
    checkCudaErrors(cudaMemset(b, 2, elements * sizeof(float)));
    checkCudaErrors(cudaMemset(c, 0, elements * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = 32 * numberOfSMs;
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        bidirectionalBandwidthBenchmark<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, elements);
        checkCudaErrors(cudaPeekAtLastError()); // Check for errors in kernel launch
    }

    checkCudaErrors(cudaDeviceSynchronize()); // Ensure warmup is complete

    // Benchmark
    unsigned long long *d_start, *d_end, *d_elapsed_time; // Device pointers
    unsigned long long h_elapsed_time; // Host copy of elapsed time
    checkCudaErrors(cudaMalloc(&d_start, sizeof(unsigned long long))); // Allocate memory for the device pointers
    checkCudaErrors(cudaMalloc(&d_end, sizeof(unsigned long long))); // Allocate memory for the device pointers
    checkCudaErrors(cudaMalloc(&d_elapsed_time, sizeof(unsigned long long))); // Allocate memory for the device pointers

    // Initialize the elapsed time to 0
    initializeElapsedTime<<<1,1>>>(d_elapsed_time);
    for (int i = 0; i < ITERATIONS; ++i) {
        getStartTime<<<1,1>>>(d_start);
        bidirectionalBandwidthBenchmark<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, elements);
        getEndTime<<<1,1>>>(d_end);
        getElapsedTime<<<1,1>>>(d_start, d_end, d_elapsed_time);
    }
    
    // synchronize the device
    checkCudaErrors(cudaDeviceSynchronize()); 

    // Copy the elapsed time to the host
    cudaMemcpy(&h_elapsed_time, d_elapsed_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // To convert cycles to time, you need the GPU clock rate (KHz) here. 
    // This is a device property, so you need to query it.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    double clockRate = deviceProp.clockRate * 1e3; // KHz to Hz
    double elapsedTimeInSecs = h_elapsed_time / clockRate;

    // Compute bandwidth in GB/s
    double total_size = (elements * sizeof(float) * 3 * ITERATIONS);
    double bandwidth = get_bandwidth(total_size, elapsedTimeInSecs);

    // Free device memory
    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));
    checkCudaErrors(cudaFree(c));
    checkCudaErrors(cudaFree(d_start));
    checkCudaErrors(cudaFree(d_end));
    checkCudaErrors(cudaFree(d_elapsed_time));

    return bandwidth;
}

double run_cacheBandwidthCopyBenchmark(size_t cache_size) {
    int elements = (cache_size / sizeof(float)) * FRACTION_COPY; // 2 arrays each of size cache_size/FRACTION
    float *a, *b;

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    // Get the number of SMs on the current GPU
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&a, elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&b, elements * sizeof(float)));

    // Use cudaMemset to initialize device memory
    checkCudaErrors(cudaMemset(a, 1, elements * sizeof(float)));
    checkCudaErrors(cudaMemset(b, 2, elements * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = 32 * numberOfSMs;
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        cacheBandwidthCopyBenchmark<<<blocksPerGrid, threadsPerBlock>>>(a, b, elements);
        checkCudaErrors(cudaPeekAtLastError()); // Check for errors in kernel launch
    }

    checkCudaErrors(cudaDeviceSynchronize()); // Ensure warmup is complete

    // Benchmark
    unsigned long long *d_start, *d_end, *d_elapsed_time; // Device pointers
    unsigned long long h_elapsed_time; // Host copy of elapsed time
    cudaMalloc(&d_start, sizeof(unsigned long long)); // Allocate memory for the device pointers
    cudaMalloc(&d_end, sizeof(unsigned long long)); // Allocate memory for the device pointers
    cudaMalloc(&d_elapsed_time, sizeof(unsigned long long)); // Allocate memory for the device pointers

    // Initialize the elapsed time to 0
    initializeElapsedTime<<<1,1>>>(d_elapsed_time);
    for (int i = 0; i < ITERATIONS; ++i) {
        getStartTime<<<1,1>>>(d_start);
        cacheBandwidthCopyBenchmark<<<blocksPerGrid, threadsPerBlock>>>(a, b, elements);
        getEndTime<<<1,1>>>(d_end);
        getElapsedTime<<<1,1>>>(d_start, d_end, d_elapsed_time);
    }
    
    // synchronize the device
    checkCudaErrors(cudaDeviceSynchronize()); 

    // Copy the elapsed time to the host
    cudaMemcpy(&h_elapsed_time, d_elapsed_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // To convert cycles to time, you need the GPU clock rate (KHz) here. 
    // This is a device property, so you need to query it.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    double clockRate = deviceProp.clockRate * 1e3; // KHz to Hz
    double elapsedTimeInSecs = h_elapsed_time / clockRate;

    // Compute bandwidth in GB/s
    double total_size = (elements * sizeof(float) * 2 * ITERATIONS);
    double bandwidth = get_bandwidth(total_size, elapsedTimeInSecs);

    // Free device memory
    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));
    checkCudaErrors(cudaFree(d_start));
    checkCudaErrors(cudaFree(d_end));
    checkCudaErrors(cudaFree(d_elapsed_time));

    return bandwidth;
}

double run_cache_benchmark(size_t cache_size) {
    return run_bidirectionalBandwidthBenchmark(cache_size);
    //return run_cacheBandwidthCopyBenchmark(cache_size);
}

#endif // CACHE_BENCHMARK_H
