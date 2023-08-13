#include <unordered_map>

#pragma once

// quick notes
// int* means integer pointer, which could also mean integer array


class QuickSort
{
public:
	int* unsorted;
	int* sorted;
	size_t size;

	QuickSort(int* arr, size_t size);
	~QuickSort();
	void sort();
	void display(); // this is where all the fancy fancy stuff gon be shown

private:
	// this is where all the inner workings will be
	void quicksort(int* arr, int low, int high);
	int partition(int* arr, int low, int high); // returns index of pivot element
	void swap(int* p1, int* p2);
};
