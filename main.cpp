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
	printf("This is not used\n");
	return 0;
}
