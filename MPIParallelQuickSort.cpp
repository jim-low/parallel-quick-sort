#include <iostream>
#include <mpi.h>
#include "MPIParallelQuickSort.h"

#pragma region
// 1. Divide the n data values into p equal parts [n/p] data values per processor.
// 2. Select the pivot element randomly on first processor p0 and breadcast it to each processor

// 3. Perform global sort
// 3.1 Locally in each processor, divide the data into two sets according to the pivot (smaller or larger)
// 3.2 Split the processors into two groups and exchange data pair wise between
// them so that all processors in one group get data less than the pivot and
// the others get data larger than the pivot.

// 4. Repeat 3.1 - 3.2 recursively for each half.
// 5. Each processor sorts the items it has, using quick sort.
#pragma endregion

MPIParallelQuickSort::MPIParallelQuickSort(float* arr, size_t size)
{
	// basic initializiation
	this->size = size;
	this->unsorted = (float*)calloc(size, sizeof(float));
	this->sorted = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		this->sorted[i] = arr[i];
	}

	// MPI Initialization
	this->rank = 0;
	this->numProcesses = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
	MPI_Comm_size(MPI_COMM_WORLD, &this->numProcesses);
}

MPIParallelQuickSort::~MPIParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}

void MPIParallelQuickSort::sort()
{

}
