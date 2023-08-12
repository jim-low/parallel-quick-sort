#include <iostream>
#include <ctime>
#include "omp.h"
#include "QuickSort.h"
#include "ParallelQuickSort.h"

int main()
{
	size_t size = 1000000;
	int* arr = (int*)calloc(size, sizeof(int));
	// int size = sizeof(arr) / sizeof(int);

	for (int i = 0; i < size; ++i)
	{
		arr[i] = -(int)size + (rand() % size);
	}

	QuickSort quickSort = QuickSort(arr, size);

	clock_t start = clock();
	quickSort.sort();
	clock_t end = clock();

	//quickSort.display();
	double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

	std::cout << "Time taken: " << elapsedTime << " seconds" << std::endl;

	return 0;
}
