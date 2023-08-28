#include <iostream>
#include <vector>
#include "omp.h"
#include "OMPParallelQuickSort.h"

#pragma region Notes

// Here are the possible pragma omp clauses that we may need to make it parallel (according to ChatGPT):
// #pragma omp parallel
// #pragma omp single nowait
// #pragma omp task
// #pragma omp taskwait
// #pragma omp parallel for

#pragma endregion Notes

OMPParallelQuickSort::OMPParallelQuickSort(int* arr, size_t size)
{
	this->size = size;
	this->unsorted = (int*)calloc(size, sizeof(int));
	this->sorted = (int*)calloc(size, sizeof(int));

	// deep copy into array
	for (int i = 0; i < size; ++i)
	{
		this->unsorted[i] = arr[i];
		//this->sorted[i] = arr[i];
	}

	// 1.0 Divide the n data values into p equal parts, [n/p] data values per processor
	this->numProcs = omp_get_num_procs();
	this->splitArray = (int**)calloc(numProcs, sizeof(int*));
	for (int i = 0; i < numProcs; ++i)
	{
		this->splitArray[i] = (int*)calloc(size / numProcs, sizeof(int));
	}

	int margin = this->size / this->numProcs;
	for (int i = 0; i < this->numProcs; ++i)
	{
		for (int j = 0; j < margin; ++j)
		{
			this->splitArray[i][j] = arr[(i * margin) + j];
		}
	}
}

OMPParallelQuickSort::~OMPParallelQuickSort()
{
	free(this->unsorted);
	free(this->sorted);

	for (int i = 0; i < this->numProcs; ++i)
	{
		free(this->splitArray[i]);
	}
	free(this->splitArray);
}

void OMPParallelQuickSort::sort()
{
	//this->quicksort(this->sorted, 0, this->size - 1);

	// 2.0  select the pivot element randomly on first processor p0 and broadcast it to each processor
	int margin = this->size / this->numProcs; // this should get the array size per processor
	int pivotElementIndex = rand() % (margin + 1);

	// 3.0 perform global sort
	this->globalSort(pivotElementIndex);
}

void OMPParallelQuickSort::display()
{
	std::cout << "Unsorted Array:" << std::endl;
	for (int i = 0; i < this->size; ++i) {
		std::cout << this->unsorted[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "Sorted Array:" << std::endl;
	for (int i = 0; i < this->size; ++i) {
		std::cout << this->sorted[i] << " ";
	}
	std::cout << std::endl;
}

void OMPParallelQuickSort::globalSort(int pivot)
{
	std::vector<int> smallerThanPivot;
	std::vector<int> largerThanPivot;
	#pragma omp parallel
	{
		// 3.1 Locally, in each processor, divide the data into two sets according to the pivot (smaller or larger)
		#pragma omp for
		for (int i = 0; i < this->numProcs; ++i)
		{
			for (int j = 0; j < (this->size / this->numProcs); ++j)
			{
				int value = this->splitArray[i][j];
				if (value < pivot)
				{
					#pragma omp critical
					smallerThanPivot.push_back(value);
				}
				else
				{
					#pragma omp critical
					largerThanPivot.push_back(value);
				}
			}
		}
	}

	// 3.2 split the processors into two groups and exchange data pair wise
	// between them so that all processors in one group get data less than the
	// pivot and the others get data larger than the pivot

	// 4.0 repeat 3.1 - 3.2 recursively for each half
}

void OMPParallelQuickSort::quicksort(int* arr, int low, int high)
{
	if (low < high)
	{
		int pivotIndex = this->partition(arr, low, high);

		this->quicksort(arr, low, pivotIndex - 1);
		this->quicksort(arr, pivotIndex + 1, high);
	}
}

// returns index of pivot
int OMPParallelQuickSort::partition(int* arr, int low, int high)
{
	int pivot = arr[high];
	int swapMarker = low - 1;

	for (int j = low; j < high; ++j) {
		if (arr[j] <= pivot) {
			++swapMarker;
			swap(&arr[swapMarker], &arr[j]);
		}
	}

	swap(&arr[swapMarker + 1], &arr[high]);
	return swapMarker + 1;
}

void OMPParallelQuickSort::swap(int* p1, int* p2)
{
	int temp = *p1;
	*p1 = *p2;
	*p2 = temp;
}
