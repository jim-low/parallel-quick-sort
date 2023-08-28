#pragma once

class CUDAParallelQuickSort
{
public:
	size_t size;
	int* unsorted; // this has the sole purpose of displaying the result of the sort
	int* sorted;

	CUDAParallelQuickSort(int* arr, size_t size);
	~CUDAParallelQuickSort();

private:
};

