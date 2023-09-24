#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>
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
		//arr[i] = ((float)rand()) / ((float)rand());
		arr[i] = size - i;
	}

	return arr;
}

int main(int argc, char** argv)
{
	srand(time(0));
	size_t size = 10000; // max = 32768
	float* arr = generate_float_array(size);

	// MPI Parallel Quick Sort
	MPI_Init(&argc, &argv);
	int rank, numProcesses;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	if (rank == 0)
	{
		std::cout << std::endl;
		std::cout << "Array size: " << size << " (descending)" << std::endl;
		std::cout << std::endl;
	}

	MPIParallelQuickSort mpiSort = MPIParallelQuickSort(arr, size, rank, numProcesses);
	auto start_time = std::chrono::high_resolution_clock::now();
	mpiSort.sort();
	auto end_time = std::chrono::high_resolution_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		//mpiSort.display();
		std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
		std::cout << "Elapsed time (MPI): " << elapsed_time.count() << " milliseconds" << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// Standard Quick Sort
	QuickSort quicksort = QuickSort(arr, size);
	start_time = std::chrono::high_resolution_clock::now();
	quicksort.sort();
	end_time = std::chrono::high_resolution_clock::now();

	if (rank == 0)
	{
		//quicksort.display();
		std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
		std::cout << "Elapsed time: " << elapsed_time.count() << " milliseconds" << std::endl;
		std::cout << std::endl;
	}

	MPI_Finalize();
	free(arr);
	return 0;
}
