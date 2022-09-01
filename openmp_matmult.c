#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include "tools/tools.h"

#define TIMES_TO_REPEAT 1

int main(int argc, char* argv[]) {
	int idim = 4000;
	int jdim = 4000;
	int kdim = 4000;
	int i, j, k, nt;
	double t1, times[TIMES_TO_REPEAT];
	float nops, err;
	float* A, *B, *C, *actualC;
	A = (float*)malloc(idim * kdim * sizeof(float));
	B = (float*)malloc(kdim * jdim * sizeof(float));
	C = (float*)malloc(idim * jdim * sizeof(float));
	actualC = (float*)malloc(idim * jdim * sizeof(float));

	// Initialize matrices
	zero_init(idim, jdim, actualC);
	zero_init(idim, jdim, C);
	rand_init(idim, kdim, A);
	rand_init(kdim, jdim, B);

	// This is the standard matrix multiplication - do not adjust
	matrix_mult(idim, jdim, kdim, A, B, actualC);

	// Begin test multiplication
#pragma omp parallel
	nt = omp_get_max_threads();
	printf("Running with %d threads\n", nt);

	for (int loop_cnt = 0; loop_cnt < TIMES_TO_REPEAT; loop_cnt++) {
		t1 = get_wall_time(); // record start time
#pragma omp parallel shared(A, B, C) private(i, j, k)
		{
#pragma omp for schedule(static)
			for (i = 0; i < idim; i++)
				for (k = 0; k < kdim; k++)
					for (j = 0; j < jdim; j++)
						C[i * jdim + j] += A[i * kdim + k] * B[k * jdim + j];
		}
		t1 = get_wall_time() - t1; // record elapsed time
		times[loop_cnt] = t1;
		if (loop_cnt != (TIMES_TO_REPEAT - 1))
			zero_init(idim, jdim, C);
	}

	// Error calculation
	err = error_calc(idim, jdim, actualC, C);

	t1 = 0.0;
	for (i = 0; i < TIMES_TO_REPEAT; i++)
		t1 += times[i];
	t1 /= (float) TIMES_TO_REPEAT;
	printf("Finished in %lf seconds\n", t1);
	t1 *= (1.e+09);
	nops = (float)2 * idim * kdim * jdim;
	printf("Performance = %f GFLOPs\n", nops / t1);
	printf("Error: %f\n", err);

	return 0;
}