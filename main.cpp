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

	//for (int i = 0; i < size; ++i)
	//{
	//	std::cout << arr[i] << " ";
	//}
	//std::cout << std::endl;
	//std::cout << std::endl;
	//std::cout << std::endl;

	// MPI Parallel Quick Sort
	//MPI_Init(&argc, &argv);
	//MPIParallelQuickSort mpiSort = MPIParallelQuickSort(arr, size);
	//mpiSort.sort();
	//MPI_Finalize();

	// OpenMP Part (need to comment the upper part to run)
	double runtime;
	omp_set_num_threads(8);

	OMPParallelQuickSort ompSort = OMPParallelQuickSort(arr, size);
	runtime = omp_get_wtime();
	ompSort.sort();
	ompSort.display();

	runtime = omp_get_wtime() - runtime;
	// std::cout << "\n\nUsed " << runtime << " seconds." << std::endl;
	printf("\nUsed %.9f seconds.\n\n", runtime);

	// Part for used to compare
	QuickSort ompSort2 = QuickSort(arr, size);
	runtime = omp_get_wtime();
	ompSort2.sort();
	ompSort2.display();

	runtime = omp_get_wtime() - runtime;

	printf("\nUsed %.9f seconds without OpenMP.\n\n", runtime);

	free(arr);
	return 0;
}
