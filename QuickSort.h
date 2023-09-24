#include <unordered_map>

#pragma once

// quick notes
// int* means integer pointer, which could also mean integer array


class QuickSort
{
public:
	float* unsorted;
	float* sorted;
	size_t size;

	QuickSort(float* arr, size_t size);
	~QuickSort();
	void sort();
	void display(); // this is where all the fancy fancy stuff gon be shown

	float* getSorted();

private:
	// this is where all the inner workings will be
	void quicksort(float* arr, int low, int high);
	float partition(float* arr, int low, int high); // returns index of pivot element
	void swap(float* p1, float* p2);
};
