#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include "MPIParallelQuickSort.h"
#include "QuickSort.h"

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

MPIParallelQuickSort::MPIParallelQuickSort(float* arr, size_t size, int rank, int numProcesses)
{
	// basic initializiation
	this->size = size;
	this->unsorted = new float[size];
	this->sorted = new float[size];

	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
	}

	// MPI Initialization
	this->rank = rank;
	this->numProcesses = numProcesses;

	this->margin = static_cast<int>(std::ceil((double)this->size / (double)this->numProcesses));
	this->half_size = this->margin / 2;

	//std::cout << "size: " << size << std::endl;
	//std::cout << "numProcesses: " << numProcesses << std::endl;
	//std::cout << "margin: " << margin << std::endl;
	//std::cout << "half_size: " << half_size << std::endl;
}

MPIParallelQuickSort::~MPIParallelQuickSort()
{
	//std::cout << "freeing stuff" << std::endl;
	free(this->unsorted);
	free(this->sorted);
}

void MPIParallelQuickSort::sort()
{
	// 1. Divide the n data values into p equal parts [n/p] data values per processor.
	float* localArray = new float[margin];
	MPI_Scatter(this->unsorted, margin, MPI_FLOAT, localArray, margin, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// 2. Select the pivot element randomly on first processor p0 and broadcast it to each processor
	float pivotElement = 0;
	if (this->rank == 0)
	{
		pivotElement = localArray[(rand() % margin)];
		MPI_Bcast(&pivotElement, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// 3. Perform global sort
	// 3.1 Locally in each processor, divide the data into two sets according to the pivot (smaller or larger)
	std::vector<float> lowerThanPivot;
	std::vector<float> higherThanPivot;
	this->splitDataInProcessors(&lowerThanPivot, &higherThanPivot, localArray, pivotElement);

	// 3.2 Split the processors into two groups and exchange data pair wise between
	// them so that all processors in one group get data less than the pivot and
	// the others get data larger than the pivot.
	this->exchangeDataInProcessors(&lowerThanPivot, &higherThanPivot);

	// 4. Each processor sorts the items it has, using quick sort.
	std::vector<float> combined;
	combined.reserve(lowerThanPivot.size() + higherThanPivot.size());
	combined.insert(combined.end(), lowerThanPivot.begin(), lowerThanPivot.end());
	combined.insert(combined.end(), higherThanPivot.begin(), higherThanPivot.end());

	QuickSort quicksort = QuickSort(&combined[0], this->margin);
	quicksort.sort();
	float* newSortedArray = quicksort.getSorted();
	MPI_Gather(newSortedArray, this->margin, MPI_FLOAT, this->sorted, this->margin, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		QuickSort finalSort = QuickSort(this->sorted, this->size);
		finalSort.sort();
		float* lastArrayToBeDeclared = finalSort.getSorted();

		for (int i = 0; i < this->size; ++i)
		{
			this->sorted[i] = lastArrayToBeDeclared[i];
		}
	}

	free(localArray);
}

void MPIParallelQuickSort::display()
{
	std::cout << "Unsorted Array (MPI):" << std::endl;
	for (int i = 0; i < this->size; ++i)
	{
		std::cout << this->unsorted[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "Sorted Array (MPI):" << std::endl;
	for (int i = 0; i < this->size; ++i)
	{
		std::cout << this->sorted[i] << " ";
	}
	std::cout << std::endl;
}

void MPIParallelQuickSort::splitDataInProcessors(std::vector<float>* lower, std::vector<float>* higher, float* arr, float pivot)
{
	for (int i = 0; i < this->margin; ++i)
	{
		if (arr[i] < pivot)
		{
			if (lower->size() >= half_size)
			{
				higher->push_back(arr[i]);
				continue;
			}
			lower->push_back(arr[i]);
		}
		else
		{
			if (higher->size() >= half_size)
			{
				lower->push_back(arr[i]);
				continue;
			}
			higher->push_back(arr[i]);
		}
	}
}

void MPIParallelQuickSort::exchangeDataInProcessors(std::vector<float>* lower, std::vector<float>* higher)
{
	float* lower_data = nullptr;
	float* higher_data = nullptr;
	float* received_data = nullptr;
	if (this->rank == 0)
	{
		higher_data = new float[half_size];
		received_data = new float[half_size];
		for (int i = 0; i < half_size; ++i)
		{
			higher_data[i] = higher->at(i);
		}

		MPI_Send(higher_data, half_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(received_data, half_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (int i = 0; i < higher->size(); ++i)
		{
			higher->at(i) = received_data[i];
		}

		free(higher_data);
		free(received_data);
	}
	else
	{
		lower_data = new float[half_size];
		higher_data = new float[half_size];
		received_data = new float[half_size];
		// receive higher array from previous processor
		MPI_Recv(received_data, half_size, MPI_FLOAT, this->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (int i = 0; i < half_size; ++i)
		{
			lower_data[i] = lower->at(i);
			higher_data[i] = higher->at(i);
		}

		// swap received array with current lower array
		for (int i = 0; i < half_size; ++i)
		{
			lower->at(i) = received_data[i];
		}

		if (this->rank != (this->numProcesses - 1)) // if current rank is not final processor rank, then copy higher and send to next processor
		{
			MPI_Send(higher_data, half_size, MPI_FLOAT, this->rank + 1, 0, MPI_COMM_WORLD); // send current higher array to next processor
		}

		MPI_Send(lower_data, half_size, MPI_FLOAT, this->rank - 1, 0, MPI_COMM_WORLD); // send current higher array to next processor

		free(lower_data);
		free(higher_data);
		free(received_data);
	}
}
