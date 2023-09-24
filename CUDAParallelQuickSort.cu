#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "CUDAParallelQuickSort.cuh"


//constructor
CUDAParallelQuickSort::CUDAParallelQuickSort(float* arr, size_t size)
{
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    cudaError_t err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24);
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSetLimit failed! Error: " << cudaGetErrorString(err) << std::endl;
        // Handle the error
        return;
    }

    this->size = size;

    this->h_unsorted = (float*)calloc(size, sizeof(float));
    this->h_sorted = (float*)calloc(size, sizeof(float));

    cudaMalloc((void**)&this->d_unsorted, size * sizeof(float));
    cudaMalloc((void**)&this->d_sorted, size * sizeof(float));

    cudaMemcpy((void*)this->d_unsorted, (void*)arr, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)this->d_sorted, (void*)arr, size * sizeof(float), cudaMemcpyHostToDevice);
}

//deconstructor
CUDAParallelQuickSort::~CUDAParallelQuickSort()
{
    free(this->h_unsorted);
    free(this->h_sorted);

    cudaFree(this->d_unsorted);
    cudaFree(this->d_sorted);
}


//CUDA kernel for partitioning
__device__ float partition(float* arr, int low, int high)
{
    float pivot = arr[high];

    int swapMarker = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            ++swapMarker;
            float temp = arr[swapMarker];
            arr[swapMarker] = arr[j];
            arr[j] = temp;
        }
    }

    float temp = arr[swapMarker + 1];
    arr[swapMarker + 1] = arr[high];
    arr[high] = temp;

    return swapMarker + 1;
}

__device__ void quickSort(float* arr, int left, int right) {

    if (left < right) {
        int pivotIndex = partition(arr, left, right);
        quickSort(arr, left, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, right);
    }
}

//global functions can be called from the host and executed on the device.
__global__ void cudaQuicksort(float* arr, int left, int right, int maxDepth) {
    
    int pivotIndex = partition(arr, left, right); //getting the pivot index and initiating the partition process

    if (left < right)
    {
        if (maxDepth > 0) {
            //cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cudaQuicksort <<<1, 8>>> (arr, left, pivotIndex - 1, maxDepth - 1);

            //cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
            cudaQuicksort <<<1, 8>>> (arr, pivotIndex + 1, right, maxDepth - 1);

            quickSort(arr, left, right);
            //printf("MAXDEPTH LEFT: %d.\n", maxDepth);
        
        }

        else {
            //printf("MAXDEPTH REACHED, RESORTING TO STANDARD SORT\n");
            quickSort(arr, left, right);
        }
        
        

    }
}

__host__ void CUDAParallelQuickSort::sort()
{
    cudaQuicksort<<<1, 8>>>(this->d_sorted, 0, size - 1, 16);
    cudaDeviceSynchronize();
}

//display the result
void CUDAParallelQuickSort::display()
{
    cudaError_t cudaStatus1 = cudaStreamSynchronize(0);
    if (cudaStatus1 != cudaSuccess) {
        printf("Stream 0 is not done yet.\n");
        return;
    }

    //cudaEventRecord(start);
    cudaError_t cudaStatus = cudaMemcpy(this->h_unsorted, this->d_unsorted, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    cudaStatus = cudaMemcpy(this->h_sorted, this->d_sorted, size * sizeof(float), cudaMemcpyDeviceToHost); 
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaMemcpy failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }


    printf("Unsorted Array:\n");
    for (int i = 0; i < this->size; ++i) {
        std::cout << this->h_unsorted[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Sorted Array:" << std::endl;
    for (int i = 0; i < this->size; ++i) {
        std::cout << this->h_sorted[i] << " ";
    }
    std::cout << std::endl;
}



