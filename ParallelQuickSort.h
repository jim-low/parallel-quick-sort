#pragma once

// as you might have expected, this is gonna be exactly like the QuickSort class, but in parallel

class ParallelQuickSort
{
public:
	ParallelQuickSort(int* unsorted_array);
	int* sort();
	void display();

private:
	// this is where all the inner workings will be
};

