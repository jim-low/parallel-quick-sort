#include <iostream>
#include <ctime>
#include "omp.h"
#include "QuickSort.h"
#include "ParallelQuickSort.h"

void performQuickSort(int* arr, size_t size)
{
	QuickSort quicksort = QuickSort(arr, size);

	clock_t start = clock();
	quicksort.sort();
	clock_t end = clock();

	//quicksort.display();
	double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

	std::cout << "Time taken to sort " << size << " elements: " << elapsedTime << " seconds" << std::endl;
}

void performParallelQuickSort(int* arr, size_t size)
{
	QuickSort quicksort = QuickSort(arr, size);

	clock_t start = clock();
	quicksort.sort();
	clock_t end = clock();

	//quicksort.display();
	double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

	std::cout << "Time taken to sort " << size << " elements with parallelism: " << elapsedTime << " seconds" << std::endl;
}

int main()
{
	size_t size = 1000000;
	int* arr = (int*)calloc(size, sizeof(int));
	//int size = sizeof(arr) / sizeof(int);

	for (int i = 0; i < size; ++i)
	{
		arr[i] = (rand() % size);
	}

	performParallelQuickSort(arr, size);
	performQuickSort(arr, size);

	std::cout << "parallel quick sort is not really parallel (probably)" << std::endl;

	free(arr);

	return 0;
}
