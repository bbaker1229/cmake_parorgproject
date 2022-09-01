#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include "tools/tools.h"

#define TIMES_TO_REPEAT 1

int main(int argc, char* argv[]) {
    int idim = 400;
    int jdim = 400;
    int kdim = 400;
    int i, j, k, rowlen, vallen, nt;
    long int newdim;
    float nops, per, err;
    double t1, times[TIMES_TO_REPEAT];
    float* A, * B, * C, * actualC;
    A = (float*)malloc(idim * kdim * sizeof(float));
    B = (float*)malloc(kdim * jdim * sizeof(float));
    C = (float*)malloc(idim * jdim * sizeof(float));
    actualC = (float*)malloc(idim * jdim * sizeof(float));

    // Initialize matrices
    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    if (argc == 1)
        per = 0.3;
    else
        per = atof(argv[1]);
    printf("Running with %0.1f%% sparsity\n", per * 100);
    newdim = make_sparse_percent(per, idim, kdim, A);
    
    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);

    int* rowval, * colval;
    float* value;
    rowval = (int*)malloc(newdim * sizeof(int));
    colval = (int*)malloc(newdim * sizeof(int));
    value = (float*)malloc(newdim * sizeof(float));
    make_sparse_matrix(idim, kdim, rowval, colval, value, &rowlen, &vallen, A);

    for (int loop_cnt = 0; loop_cnt < TIMES_TO_REPEAT; loop_cnt++) {
#pragma omp parallel
        nt = omp_get_num_threads();
        printf("Running with %d threads\n", nt);

        t1 = get_wall_time();
#pragma omp parallel for
        for (i = 0; i < rowlen - 1; i++)
            for (k = rowval[i]; k < rowval[i + 1]; k++)
                for (j = 0; j < jdim; j++)
                    C[i * jdim + j] += value[k] * B[colval[k] * jdim + j];
        t1 = get_wall_time() - t1;
        times[loop_cnt] = t1;
        if (loop_cnt != (TIMES_TO_REPEAT - 1))
            zero_init(idim, jdim, C);
    }

    // error calculation
    // TODO: Fix the error.  It is not correct.
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
    return(0);
}