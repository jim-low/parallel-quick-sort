#pragma once

class OMPParallelQuickSort
{
public:
	size_t size;
	int* unsorted; // this has the sole purpose of displaying the result of the sort
	int* sorted;

	OMPParallelQuickSort(int* arr, size_t size);
	~OMPParallelQuickSort();

private:
};

