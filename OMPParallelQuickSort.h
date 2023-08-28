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
	void sort();
	void display(); // this is where all the fancy fancy stuff gon be shown

private:
	// this is where all the inner workings will be
	void globalSort(int pivot);
	void quicksort(int* arr, int low, int high);
	int partition(int* arr, int low, int high); // returns index of pivot element
	void swap(int* p1, int* p2);
};

