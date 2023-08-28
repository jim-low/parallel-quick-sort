#include <iostream>
#include "CUDAParallelQuickSort.h"

CUDAParallelQuickSort::CUDAParallelQuickSort(float* arr, size_t size)
{
	this->size = size;
	this->unsorted = (float*)calloc(size, sizeof(float));
	this->sorted = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		this->sorted[i] = arr[i];
	}
}

CUDAParallelQuickSort::~CUDAParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}
