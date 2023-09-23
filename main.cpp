#include <iostream>
#include <ctime>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include "QuickSort.h"
#include "OMPParallelQuickSort.h"
#include "MPIParallelQuickSort.h"
#include "CUDAParallelQuickSort.h"

float* generate_float_array(size_t size)
{
	float* arr = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < size; ++i)
	{
		arr[i] = ((float)rand()) / ((float)rand());
	}

	return arr;
}

int main(int argc, char** argv)
{
	srand(time(0));
	size_t size = 16;
	float* arr = generate_float_array(size);

	for (int i = 0; i < size; ++i)
	{
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	// MPI Parallel Quick Sort
	MPI_Init(&argc, &argv);
	MPIParallelQuickSort mpiSort = MPIParallelQuickSort(arr, size);
	mpiSort.sort();
	MPI_Finalize();

	// OpenMP Part (need to comment the upper part to run)
	/*omp_set_num_threads(4);

	OMPParallelQuickSort ompSort = OMPParallelQuickSort(arr, size);
	ompSort.sort();

	for (int i = 0; i < size; ++i)
	{
		std::cout << ompSort.sorted[i] << " ";
	}
	std::cout << std::endl;*/

	free(arr);
	return 0;
}
