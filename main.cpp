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
		arr[i] = -800 + (rand() % 1000000);
	}

	QuickSort quickSort = QuickSort(arr, size);

	clock_t start = clock();
	quickSort.sort();
	clock_t end = clock();

	quickSort.display();
	//std::cout << "it took too much time to print out 1 million numbers so heres a line of text for reference." << std::endl;
	double elapsedTime = double(end - start) / CLOCKS_PER_SEC;

	std::cout << "Time taken: " << elapsedTime << " seconds" << std::endl;

	return 0;
}
