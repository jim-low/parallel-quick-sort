#pragma once

class OMPParallelQuickSort
{
public:
	size_t size;
	float* unsorted; // this has the sole purpose of displaying the result of the sort
	float* sorted;

	OMPParallelQuickSort(float* arr, size_t size);
	~OMPParallelQuickSort();

private:
};

