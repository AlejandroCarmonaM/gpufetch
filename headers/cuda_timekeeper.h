/*Header file for functions to measure time in cycles in CUDA*/

#ifndef CUDA_TIMEKEEPER_H
#define CUDA_TIMEKEEPER_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // TIME KEEPING
    //  Kernel to capture the start time
    __global__ void getStartTime(unsigned long long *start)
    {
        if (threadIdx.x == 0)
        {
            *start = clock64();
        }
    }

    // Kernel to capture the end time
    __global__ void getEndTime(unsigned long long *end)
    {
        if (threadIdx.x == 0)
        {
            *end = clock64();
        }
    }

    // Kernel to get the elapsed time
    __global__ void getElapsedTime(unsigned long long *start, unsigned long long *end, unsigned long long *elapsedTime)
    {
        if (threadIdx.x == 0)
        {
            *elapsedTime += *end - *start;
        }
    }

    // Kernel to initialize the elapsed time to 0
    __global__ void initializeElapsedTime(unsigned long long *elapsedTime)
    {
        if (threadIdx.x == 0)
        {
            *elapsedTime = 0;
        }
    }

    //Returns the bandwidth in GB/s
    double get_bandwidth(double total_size, double elapsedTimeInSecs)
    {
        double bandwidth = (total_size / (elapsedTimeInSecs)) / (1<<30);
        return bandwidth;
    }

    

#ifdef __cplusplus
}
#endif

#endif // CUDA_TIMEKEEPER_H