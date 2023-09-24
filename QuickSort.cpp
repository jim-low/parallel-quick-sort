#include <iostream>
#include "QuickSort.h"

QuickSort::QuickSort(float* arr, size_t size)
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

QuickSort::~QuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}

void QuickSort::sort()
{
	this->quicksort(this->sorted, 0, this->size - 1);
}

void QuickSort::display()
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

float* QuickSort::getSorted()
{
	return this->sorted;
}

void QuickSort::quicksort(float* arr, int low, int high)
{
	if (low < high)
	{
		int pivotIndex = this->partition(arr, low, high);

		this->quicksort(arr, low, pivotIndex - 1);
		this->quicksort(arr, pivotIndex + 1, high);
	}
}

// returns index of pivot
float QuickSort::partition(float* arr, int low, int high)
{
	float pivot = arr[high];
	int swapMarker = low - 1; // idk why but fuck it, but it works

	for (int j = low; j < high; ++j) {
		if (arr[j] <= pivot) {
			++swapMarker;
			swap(&arr[swapMarker], &arr[j]);
		}
	}

	swap(&arr[swapMarker + 1], &arr[high]);
	return swapMarker + 1;
}

void QuickSort::swap(float* p1, float* p2)
{
	float temp = *p1;
	*p1 = *p2;
	*p2 = temp;
}
