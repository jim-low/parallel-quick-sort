#pragma once

// as you might have expected, this is gonna be exactly like the QuickSort class, but in parallel

class OMPParallelQuickSort
{
public:
	int* unsorted;
	int* sorted;
	size_t size;

	int** splitArray;
	int numProcs;

	OMPParallelQuickSort(int* arr, size_t size);
	~OMPParallelQuickSort();

private:
};

