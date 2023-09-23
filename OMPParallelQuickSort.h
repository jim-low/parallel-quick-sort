#pragma once

class OMPParallelQuickSort
{
public:
	float* sorted;

	OMPParallelQuickSort(float* arr, size_t size);
	~OMPParallelQuickSort();

	void sort();
	void display();

private:
	size_t size;
	float* unsorted; // this has the sole purpose of displaying the result of the sort

	void quicksort(float* arr, int low, int high);
	int partition(float* arr, int low, int high);
	void swap(float* p1, float* p2);
};

