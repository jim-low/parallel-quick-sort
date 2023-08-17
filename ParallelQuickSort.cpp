#include <iostream>
#include "omp.h"
#include "ParallelQuickSort.h"

#pragma region Notes

// Here are the possible pragma omp clauses that we may need to make it parallel (according to ChatGPT):
// #pragma omp parallel
// #pragma omp single nowait
// #pragma omp task
// #pragma omp taskwait
// #pragma omp parallel for

#pragma endregion Notes

ParallelQuickSort::ParallelQuickSort(int* arr, size_t size)
{
	this->size = size;
	this->unsorted = (int*)calloc(size, sizeof(int));
	this->sorted = (int*)calloc(size, sizeof(int));

	// deep copy into array
	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		this->sorted[i] = arr[i];
	}
}

ParallelQuickSort::~ParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}

void ParallelQuickSort::sort()
{
	this->quicksort(this->sorted, 0, this->size - 1);
}

void ParallelQuickSort::display()
{
	std::cout << "Unsorted Array:" << std::endl;
	for (int i = 0; i < this->size; ++i) {
		std::cout << this->unsorted[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "Sorted Array:" << std::endl;
	for (int i = 0; i < this->size; ++i) {
		std::cout << this->sorted[i] << " ";
	}
	std::cout << std::endl;
}

void ParallelQuickSort::quicksort(int* arr, int low, int high)
{
	if (low < high)
	{
		int pivotIndex = 0;

		pivotIndex = this->partition(arr, low, high);

		this->quicksort(arr, low, pivotIndex - 1);
		this->quicksort(arr, pivotIndex + 1, high);
	}
}

// returns index of pivot
int ParallelQuickSort::partition(int* arr, int low, int high)
{
	int pivot = arr[high];
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

void ParallelQuickSort::swap(int* p1, int* p2)
{
	int temp = *p1;
	*p1 = *p2;
	*p2 = temp;
}
