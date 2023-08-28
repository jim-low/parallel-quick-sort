#include <iostream>
#include <vector>
#include "omp.h"
#include "OMPParallelQuickSort.h"

#pragma region Notes

// Here are the possible pragma omp clauses that we may need to make it parallel (according to ChatGPT):
// #pragma omp parallel
// #pragma omp single nowait
// #pragma omp task
// #pragma omp taskwait
// #pragma omp parallel for

#pragma endregion Notes

OMPParallelQuickSort::OMPParallelQuickSort(float* arr, size_t size)
{
	this->size = size;
	this->unsorted = (float*)calloc(size, sizeof(float));
	this->sorted = (float*)calloc(size, sizeof(float));

	// deep copy into array
	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		this->sorted[i] = arr[i];
	}
}

OMPParallelQuickSort::~OMPParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}
