#include <iostream>
#include <ctime>
#include <vector>
#include <mpi.h>
#include "omp.h"
#include "QuickSort.h"
#include "OMPParallelQuickSort.h"
#include "MPIParallelQuickSort.h"
#include "CUDAParallelQuickSort.h"

float* generate_float_array(size_t size)
{
	float* arr = (float*)calloc(size, sizeof(float));

	return arr;
}

int main(int argc, char** argv)
{
	size_t size = 10'000;
	float* arr = generate_float_array(size);

	// MPI Parallel Quick Sort
	MPI_Init(&argc, &argv);
	MPIParallelQuickSort mpiSort = MPIParallelQuickSort(arr, size);
	mpiSort.sort();
	MPI_Finalize();

	free(arr);
	return 0;
}
