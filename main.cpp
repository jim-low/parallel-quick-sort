#include <iostream>
#include "omp.h"
#include "QuickSort.h"
#include "ParallelQuickSort.h"

int main()
{
	omp_set_num_threads(420);
#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		printf("Crush my head with thighs pls from ID %d\n", ID);
	}
	return 0;
}