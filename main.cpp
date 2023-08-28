#include <iostream>
#include <ctime>
#include <vector>
#include "omp.h"
#include "QuickSort.h"
#include "OMPParallelQuickSort.h"

void performQuickSort(int* arr, size_t size)
{
	QuickSort quicksort = QuickSort(arr, size);

	clock_t start = clock();
	quicksort.sort();
	clock_t end = clock();

	//quicksort.display();
	double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

	//std::cout << "Time taken to sort " << size << " elements: " << elapsedTime << " seconds" << std::endl;
}

void performParallelQuickSort(int* arr, size_t size)
{
	OMPParallelQuickSort quicksort = OMPParallelQuickSort(arr, size);

	clock_t start = clock();
	quicksort.sort();
	clock_t end = clock();

	//quicksort.display();
	double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

	std::cout << "Time taken to sort " << size << " elements with parallelism: " << elapsedTime << " seconds" << std::endl;
}

int main()
{
	srand(time(NULL));

	int size = 10'000;
	int* arr = (int*)calloc(size, sizeof(int));

	int min = 0;
	int max = size;
	for (int i = 0; i < size; ++i)
	{
		int value = min + rand() % (max - min + 1);
		arr[i] = value;
	}

	performParallelQuickSort(arr, size);

	return 0;
}
