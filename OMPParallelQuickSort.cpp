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

void OMPParallelQuickSort::sort()
{
#pragma omp parallel
	{
#pragma omp single nowait
		quicksort(this->sorted, 0, this->size - 1);
	}
}

void OMPParallelQuickSort::quicksort(float* arr, int low, int high)
{
	if (low < high)
	{
		int pivotIndex = partition(arr, low, high);

#pragma omp task
		quicksort(arr, low, pivotIndex - 1);

#pragma omp task
		quicksort(arr, pivotIndex + 1, high);
	}
}

// returns index of pivot
int OMPParallelQuickSort::partition(float* arr, int low, int high)
{
	float pivot = arr[high];
	int swapMarker = low - 1;

	for (int j = low; j < high; ++j) {
		if (arr[j] <= pivot) {
			++swapMarker;
			swap(&arr[swapMarker], &arr[j]);
		}
	}

	swap(&arr[swapMarker + 1], &arr[high]);
	return swapMarker + 1;
}

void OMPParallelQuickSort::swap(float* p1, float* p2)
{
	float temp = *p1;
	*p1 = *p2;
	*p2 = temp;
}