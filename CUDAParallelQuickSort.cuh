#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float partition(float* arr, int low, int high);
__global__ void cudaQuicksort(float* arr, int left, int right);

class CUDAParallelQuickSort
{
public:
    size_t size;
    float* h_unsorted;
    float* d_unsorted;
    float* h_sorted;
    float* d_sorted;

    CUDAParallelQuickSort(float* arr, size_t size);
    ~CUDAParallelQuickSort();

    void display();
    void sort();
};




