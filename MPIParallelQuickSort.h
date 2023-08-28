#pragma once

class MPIParallelQuickSort
{
public:
	size_t size;
	int* unsorted; // this has the sole purpose of displaying the result of the sort
	int* sorted;

	MPIParallelQuickSort(int* arr, size_t size);
	~MPIParallelQuickSort();

private:
};

