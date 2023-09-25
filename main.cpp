#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <omp.h>
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
	//MPI_Init(&argc, &argv);
	//int rank, numProcesses;
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	//MPIParallelQuickSort mpiSort = MPIParallelQuickSort(arr, size, rank, numProcesses);
	//auto start_time = std::chrono::high_resolution_clock::now();
	//mpiSort.sort();
	//auto end_time = std::chrono::high_resolution_clock::now();
	//MPI_Barrier(MPI_COMM_WORLD);
	//if (rank == 0)
	//{
	//	//mpiSort.display();
	//	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
	//	std::cout << "Elapsed time (MPI): " << elapsed_time.count() << " milliseconds" << std::endl;
	//}
	//MPI_Barrier(MPI_COMM_WORLD);

	//OMPParallelQuickSort ompSort = OMPParallelQuickSort(arr, size);
	//runtime = omp_get_wtime();
	//ompSort.sort();
	//ompSort.display();

	//runtime = omp_get_wtime() - runtime;
	// std::cout << "\n\nUsed " << runtime << " seconds." << std::endl;
	//printf("\nUsed %.9f seconds.\n\n", runtime);

	// Part for used to compare
	//QuickSort ompSort2 = QuickSort(arr, size);
	//runtime = omp_get_wtime();
	//ompSort2.sort();
	//ompSort2.display();

	//runtime = omp_get_wtime() - runtime;

	//printf("\nUsed %.9f seconds without OpenMP.\n\n", runtime);


	//CUDA
	size_t free_memory, total_memory;
	cudaError_t cudaStatus = cudaMemGetInfo(&free_memory, &total_memory);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMemGetInfo failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
		// Handle the error
	}
	else {
		std::cout << "CUDA: Total GPU Memory: " << total_memory / (1024 * 1024) << " MB" << std::endl;
		std::cout << "CUDA: Free GPU Memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
	}

	size_t data_size_bytes = size * sizeof(float);
	double data_size_mb = static_cast<double>(data_size_bytes) / (1024 * 1024);
	std::cout << "CUDA: Requirement to sort array: " << data_size_mb << " MB" << std::endl;


	//CUDA Parallel Quick Sort
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

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("CUDA took: %.2f ms\n", milliseconds);

		CUDAsorter.display();

	}
	else {
		std::cerr << "Error: Failed to allocate memory for arr." << std::endl;
	}

	MPI_Finalize();
	free(arr);
	return 0;
}
