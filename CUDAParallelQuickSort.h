#pragma once

class CUDAParallelQuickSort
{
public:
	size_t size;
	float* unsorted; // this has the sole purpose of displaying the result of the sort
	float* sorted;

	CUDAParallelQuickSort(float* arr, size_t size);
	~CUDAParallelQuickSort();

private:
};

