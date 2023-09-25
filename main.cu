#include <iostream>
#include <ctime>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include "QuickSort.h"
#include "OMPParallelQuickSort.h"
#include "MPIParallelQuickSort.h"
#include "CUDAParallelQuickSort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float* generate_float_array(size_t size)
{
	float* arr = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < size; ++i)
	{
		arr[i] = ((float)rand()) / ((float)rand());
	}

	return arr;
}

float* generate_descending_array(size_t size) 
{
	int count;
	float* arr = (float*)calloc(size, sizeof(float));
	count = size;
	for (int i = 0; i < size; ++i)
	{
		count--;
		arr[i] = count;
	}
	return arr;
}

void standardQuicksortTest(float* arr, int size, bool display = false)
{
	QuickSort quicksort = QuickSort(arr, size);
	auto start_time = std::chrono::high_resolution_clock::now();
	quicksort.sort();
	auto end_time = std::chrono::high_resolution_clock::now();

	if (display)
	{
		quicksort.display();
	}

	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
	printf("Elapsed Time (Standard Sort): %.2f ms\n", elapsed_time);
}

void OMPSortTest(float* arr, int size, bool display = false)
{
	OMPParallelQuickSort ompSort = OMPParallelQuickSort(arr, size);
	float start_time = omp_get_wtime();
	ompSort.sort();
	float end_time = omp_get_wtime();

	if (display)
	{
		ompSort.display();
	}
	float elapsed_time = end_time - start_time;
	printf("Elapsed Time (OpenMP): %.2f ms\n", elapsed_time);

}

void MPISortTest(int* argc, char*** argv, float* arr, int size, int* rank, int* numProcesses, bool display = false)
{
	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, numProcesses);

	if (*numProcesses <= 1)
	{
		printf("ERROR: This program needs at least 2 processors to execute.");
		MPI_Finalize();
		return;
	}

	MPIParallelQuickSort mpiSort = MPIParallelQuickSort(arr, size, *rank, *numProcesses);
	auto start_time = std::chrono::high_resolution_clock::now();
	mpiSort.sort();
	auto end_time = std::chrono::high_resolution_clock::now();

	MPI_Finalize();

	if (*rank == 0)
	{
		if (display)
		{
			mpiSort.display();
		}
		std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
		printf("Elapsed Time (MPI Sort): %.2f ms\n", elapsed_time);
	}
}

void CUDAQuicksortTest(float* arr, int size, bool display = false)
{
	if (arr != nullptr) {

		QuickSort standardSorter(arr, size);

		CUDAParallelQuickSort CUDAsorter(arr, size);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		CUDAsorter.sort();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		if (display)
		{
			CUDAsorter.display();
		}

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("CUDA Sort Duration: %.2f ms\n", milliseconds);
		
	}
	else {
		std::cerr << "Error: Failed to allocate memory for arr." << std::endl;
	}
}

int main(int argc, char** argv)
{
	srand(time(0));
	size_t size = 10;
	float* arr = generate_float_array(size);

	int rank = 0;
	int numProcesses = 0;

	MPISortTest(&argc, &argv, arr, size, &rank, &numProcesses);

	if (rank == 0)
	{
		standardQuicksortTest(arr, size);
		std::cout << std::endl;
		OMPSortTest(arr, size);
		std::cout << std::endl;
		CUDAQuicksortTest(arr, size);
	}

	free(arr);
	return 0;
}
