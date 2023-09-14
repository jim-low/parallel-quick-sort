#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include "MPIParallelQuickSort.h"

#pragma region
// 1. Divide the n data values into p equal parts [n/p] data values per processor.
// 2. Select the pivot element randomly on first processor p0 and breadcast it to each processor

// 3. Perform global sort
// 3.1 Locally in each processor, divide the data into two sets according to the pivot (smaller or larger)
// 3.2 Split the processors into two groups and exchange data pair wise between
// them so that all processors in one group get data less than the pivot and
// the others get data larger than the pivot.

// 4. Repeat 3.1 - 3.2 recursively for each half.
// 5. Each processor sorts the items it has, using quick sort.
#pragma endregion

struct CustomVector
{
	int size;
	std::vector<float> data;
};

MPIParallelQuickSort::MPIParallelQuickSort(float* arr, size_t size)
{
	// basic initializiation
	this->size = size;
	this->unsorted = (float*)calloc(size, sizeof(float));
	this->sorted = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		this->sorted[i] = arr[i];
	}

	// MPI Initialization
	this->rank = 0;
	this->numProcesses = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
	MPI_Comm_size(MPI_COMM_WORLD, &this->numProcesses);
}

MPIParallelQuickSort::~MPIParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);
}

void MPIParallelQuickSort::sort()
{
	// 1. Divide the n data values into p equal parts [n/p] data values per processor.
	int margin = static_cast<int>(std::ceil((double)this->size / (double)this->numProcesses));
	float* localArray = (float*)calloc(margin, sizeof(float));
	for (int i = 0; i < margin; ++i)
	{
		localArray[i] = this->sorted[(margin * this->rank) + i];
	}

	// 2. Select the pivot element randomly on first processor p0 and broadcast it to each processor
	float pivotElement = 0;
	if (this->rank == 0)
	{
		pivotElement = localArray[(rand() % margin)];
		std::cout << pivotElement << std::endl;
		MPI_Bcast(&pivotElement, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	// 3. Perform global sort
	// 3.1 Locally in each processor, divide the data into two sets according to the pivot (smaller or larger)
	int half_size = this->size / 2;
	std::vector<float> lowerThanPivot;
	std::vector<float> higherThanPivot;
	for (int i = 0; i < this->size; ++i)
	{
		if (localArray[i] < pivotElement)
		{
			if (lowerThanPivot.size() > half_size)
			{
				higherThanPivot.push_back(localArray[i]);
				continue;
			}
			lowerThanPivot.push_back(localArray[i]);
		}
		else
		{
			if (higherThanPivot.size() > half_size)
			{
				lowerThanPivot.push_back(localArray[i]);
				continue;
			}
			higherThanPivot.push_back(localArray[i]);
		}
	}


	printf("This is before the 3.2 shit for process %d:\n", this->rank);
	for (int i = 0; i < lowerThanPivot.size(); ++i)
	{
		printf("%f ", lowerThanPivot.at(i));
	}
	for (int i = 0; i < higherThanPivot.size(); ++i)
	{
		printf("%f ", higherThanPivot.at(i));
	}
	printf("\n");

	// 3.2 Split the processors into two groups and exchange data pair wise between
	// them so that all processors in one group get data less than the pivot and
	// the others get data larger than the pivot.

	// create custom MPI datatype first to pass the vector data type across processors
	CustomVector higherVector;
	MPI_Win higherArrayWin;
	MPI_Win_create(&higherThanPivot, sizeof(std::vector<float>), sizeof(std::vector<float>), MPI_INFO_NULL, MPI_COMM_WORLD, &higherArrayWin);

	// this fucking shit i swear
	if (this->rank == 0)
	{
		higherVector.size = higherThanPivot.size();
		higherVector.data = higherThanPivot;

		// create MPI datatype for custom structure
		MPI_Datatype customVectorType;
		int block_lengths[2] = { 1, higherVector.size };
		MPI_Aint displacements[2] = { 0, sizeof(float) };
		MPI_Datatype types[2] = { MPI_FLOAT, MPI_FLOAT };
		MPI_Type_create_struct(2, block_lengths, displacements, types, &customVectorType);
		MPI_Type_commit(&customVectorType);

		MPI_Send(&higherVector, 1, customVectorType, 1, 0, MPI_COMM_WORLD); // send custom vector to processor 1
		MPI_Type_free(&customVectorType); // free custom data type
	}
	else
	{
		// receive custom vector in next processor
		MPI_Recv(&higherVector, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// swap received vector with local lowerThanPivot vector
		lowerThanPivot.swap(higherVector.data);

		higherVector.size = higherThanPivot.size();
		higherVector.data = higherThanPivot;

		// create MPI datatype for custom structure
		MPI_Datatype customVectorType;

		// higher than pivot
		int block_lengths[2] = { 1, higherVector.size };
		MPI_Aint displacements[2] = { 0, sizeof(float) };
		MPI_Datatype types[2] = { MPI_FLOAT, MPI_FLOAT };
		MPI_Type_create_struct(2, block_lengths, displacements, types, &customVectorType);
		MPI_Type_commit(&customVectorType);

		MPI_Send(&higherVector, 1, customVectorType, 1, 0, MPI_COMM_WORLD); // send custom vector to processor 1
		MPI_Type_free(&customVectorType); // free custom data type
	}

	// 4. Repeat 3.1 - 3.2 recursively for each half.
	// how the fuck to repeat???????????????????????

	// 5. Each processor sorts the items it has, using quick sort.

	free(localArray);
}
