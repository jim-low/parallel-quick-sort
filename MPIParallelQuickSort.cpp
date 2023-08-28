#include <iostream>
#include "MPIParallelQuickSort.h"

MPIParallelQuickSort::MPIParallelQuickSort(int* arr, size_t size)
{
	this->size = size;
	this->unsorted = (int*)calloc(size, sizeof(int));
	this->sorted = (int*)calloc(size, sizeof(int));

	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		this->sorted[i] = arr[i];
	}
}

MPIParallelQuickSort::~MPIParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}
