#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include "tools/tools.h"

#define TIMES_TO_REPEAT 20

int main(int argc, char* argv[]) {
	int idim = 4000;
	int jdim = 4000;
	int kdim = 4000;
	int i, j, k, p, nloc, counter;
	double t1 = 0.0;
	float nops, err;
	float* A, *B, *C, *actualC, *myA, *myC, *sendMe;
	MPI_Status status;
	A = (float*)malloc(idim * kdim * sizeof(float));
	B = (float*)malloc(kdim * jdim * sizeof(float));
	C = (float*)malloc(idim * jdim * sizeof(float));
	actualC = (float*)malloc(idim * jdim * sizeof(float));

	// Initialize matrices
	zero_init(idim, jdim, actualC);
	zero_init(idim, jdim, C);
    // rand_init(idim, kdim, A);
    // rand_init(kdim, jdim, B);
    seq_init(idim, kdim, A);
    seq_init(kdim, jdim, B);

	// This is the standard matrix multiplication - do not adjust
	matrix_mult(idim, jdim, kdim, A, B, actualC);

	// Begin test multiplication
    int myid, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        nloc = (int)(idim + nprocs - 1) / nprocs;
        myA = (float*)malloc(nloc * kdim * sizeof(float));
        sendMe = (float*)malloc(nloc * kdim * sizeof(float));
        for (i = 0; i < nloc; i++) {
            for (k = 0; k < kdim; k++) {
                myA[i * kdim + k] = A[i * kdim + k];
            }
        }

        MPI_Bcast(B, kdim * jdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (p = 1; p < nprocs; p++) {
            counter = 0;
            for (i = nloc * p; (i < nloc * (p + 1)) && (i < idim); i++) {
                j = i - nloc * p;
                for (k = 0; k < kdim; k++) {
                    sendMe[j * kdim + k] = A[i * kdim + k];
                    counter++;
                }
            }
            MPI_Send(sendMe, counter, MPI_FLOAT, p, p, MPI_COMM_WORLD);
        }
        t1 = get_wall_time();
        myC = (float*)malloc(nloc * jdim * sizeof(float));
        zero_init(nloc, jdim, myC);
        matrix_mult(nloc, jdim, kdim, myA, B, myC);
        for (i = 0; i < nloc; i++) {
            for (j = 0; j < jdim; j++) {
                C[i * jdim + j] = myC[i * jdim + j];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (p = 1; p < nprocs; p++) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &counter);
            MPI_Recv(myC, counter, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
            counter = (int)counter / jdim;
            for (i = 0; i < counter; i++) {
                k = i + nloc * status.MPI_TAG;
                for (j = 0; j < jdim; j++) {
                    C[k * jdim + j] = myC[i * jdim + j];
                }
            }
        }
        t1 = get_wall_time() - t1;

        // error calculation
        err = error_calc(idim, jdim, actualC, C);

        printf("Finished in %lf seconds\n", t1);
        t1 *= (1.e+09);
        nops = (float)2 * idim * kdim * jdim;
        printf("Performance = %f GFLOPs\n", nops / t1);
        printf("Error: %f\n", err);
    }
    else {
        MPI_Bcast(B, kdim * jdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        nloc = (int)(idim + nprocs - 1) / nprocs;
        if (myid == (nprocs - 1))
            nloc = idim - nloc * (nprocs - 1);
        myA = (float*)malloc(nloc * kdim * sizeof(float));
        zero_init(nloc, kdim, myA);
        MPI_Recv(myA, nloc * kdim, MPI_FLOAT, 0, myid, MPI_COMM_WORLD, &status);
        myC = (float*)malloc(nloc * jdim * sizeof(float));
        zero_init(nloc, jdim, myC);
        matrix_mult(nloc, jdim, kdim, myA, B, myC);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(myC, nloc * jdim, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return(0);
}