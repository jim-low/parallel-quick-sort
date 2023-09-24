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
__global__ void cudaQuicksort(float* arr, int left, int right) {
    __syncthreads();

    int pivotIndex = partition(arr, left, right); //getting the pivot index and initiating the partition process

    cudaStream_t anotherFuckingStream;
    cudaStreamCreateWithFlags(&anotherFuckingStream, cudaStreamNonBlocking);

    if (left < pivotIndex - 1)
    {
        cudaQuicksort << <1, 8, 0, anotherFuckingStream >> > (arr, left, pivotIndex - 1);
    }

    if (right > pivotIndex + 1)
    {
        cudaQuicksort << <1, 8, 0, anotherFuckingStream >> > (arr, pivotIndex + 1, right);
    }

    cudaStreamDestroy(anotherFuckingStream);
}

__host__ void CUDAParallelQuickSort::sort()
{
    cudaStream_t mainFuckingStream; //initializing stream

    //preset cudaStream, initilaize s1 to the new Stream, Set flag to be able to overlap other streams
    cudaStreamCreateWithFlags(&mainFuckingStream, cudaStreamNonBlocking);

    cudaQuicksort << <1, 8, 0, mainFuckingStream >> > (this->d_sorted, 0, size - 1);

    cudaStreamSynchronize(mainFuckingStream);

    cudaStreamDestroy(mainFuckingStream);
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



