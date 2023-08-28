#pragma once

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

class MPIParallelQuickSort
{
public:
	size_t size;
	float* unsorted; // this has the sole purpose of displaying the result of the sort
	float* sorted;

	MPIParallelQuickSort(float* arr, size_t size);
	~MPIParallelQuickSort();

private:
};

