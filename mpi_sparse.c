#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "tools/tools.h"

#define TIMES_TO_REPEAT 10

int main(int argc, char* argv[]) {
    int idim = 1000;
    int jdim = 1000;
    int kdim = 1000;
    int i, j, k, p, nloc, counter, rowlen, vallen;
    long int newdim;
    double t1 = 0.0;
    //int *rowval, *colval;
    //float *value;
    float nops, per, err;
    float* A, * B, * C, * actualC, * myB, * myC, * sendMe;
    MPI_Status status;
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
    //printf("Running with %0.1f%% sparsity\n", per * 100);
    newdim = make_sparse_percent(per, idim, kdim, A);
    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);

    int* rowval, * colval;
    float* value;
    rowval = (int*)malloc(newdim * sizeof(int));
    colval = (int*)malloc(newdim * sizeof(int));
    value = (float*)malloc(newdim * sizeof(float));
    make_sparse_matrix(idim, kdim, rowval, colval, value, &rowlen, &vallen, A);

    //    #pragma omp parallel
    int myid, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        //if(argc == 1)
        //  per = 0.3;
        //else
        //  per = atof(argv[1]);
        printf("Running with %0.1f%% sparsity\n", per * 100);
        //newdim = make_sparse_percent(per, idim,  kdim, A);
        // This is the standard matrix multiplication - do not adjust
        //matrix_mult(idim, jdim, kdim, A, B, actualC);

        //rowval = (int*) malloc(newdim*sizeof(int));
        //colval = (int*) malloc(newdim*sizeof(int));
        //value = (float*) malloc(newdim*sizeof(float));
        //make_sparse_matrix(idim, kdim, rowval, colval, value, &rowlen, &vallen, A);

        //printf("A matrix sample: \n");
        //print_sample(idim, kdim, A, 2, 10);
        //printf("B matrix sample: \n");
        //print_sample(kdim, jdim, B, 2, 10);
        //printf("actualC matrix sample: \n");
        //print_sample(idim, jdim, actualC, 2, 10);
        //C = actualC;
        printf("Running with %d procs.\n", nprocs);
        //t1 = wctime();
        nloc = (int)(jdim + nprocs - 1) / nprocs;
        //printf("proc %d using nloc of %d.\n", myid, nloc);
        myB = (float*)malloc(nloc * kdim * sizeof(float));
        sendMe = (float*)malloc(nloc * kdim * sizeof(float));
        for (i = 0; i < kdim; i++) {
            for (k = 0; k < nloc; k++) {
                myB[i * nloc + k] = B[i * jdim + k];
            }
        }
        //printf("Starting to send to other procs:\n");
        MPI_Bcast(rowval, newdim, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(colval, newdim, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(value, newdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //printf("nloc = %d.\n", nloc);
        for (p = 1; p < nprocs; p++) {
            counter = 0;
            //printf("Prepare proc %d.\n", p);
            for (i = nloc * p; (i < nloc * (p + 1)) && (i < jdim); i++) {
                j = i - nloc * p;
                //printf("Value of j = %d\n", j);
                for (k = 0; k < kdim; k++) {
                    sendMe[k * nloc + j] = B[k * jdim + i];
                    counter++;
                }
            }
            //printf("Sending...\n");
            MPI_Send(sendMe, counter, MPI_FLOAT, p, p, MPI_COMM_WORLD);
            //printf("Sent data to proc %d.\n", p);
        }
        //printf("Calculate data...\n");
        t1 = get_wall_time();
        myC = (float*)malloc(idim * nloc * sizeof(float));
        for (i = 0; i < rowlen - 1; i++)
            for (k = rowval[i]; k < rowval[i + 1]; k++)
                for (j = 0; j < nloc; j++)
                    myC[i * nloc + j] += value[k] * myB[colval[k] * nloc + j];
        //matrix_mult(nloc, jdim, kdim, myA, B, myC);
        for (i = 0; i < nloc; i++) {
            for (j = 0; j < idim; j++) {
                C[j * jdim + i] = myC[j * nloc + i];
            }
        }
        //MPI_Barrier(MPI_COMM_WORLD);
        //printf("Starting to recieve data...\n");
        for (p = 1; p < nprocs; p++) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &counter);
            //printf("found counts from proc %d using tag %d.\n", status.MPI_SOURCE, status.MPI_TAG);
            MPI_Recv(myC, counter, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
            //printf("recieved data from proc %d, in %d counts.\n", status.MPI_SOURCE, counter);
            counter = (int)counter / idim;
            //printf("nloc = %d.\n", counter);
            //print_sample(idim, counter, myC, 2, 10);
            for (i = 0; i < counter; i++) {
                k = i + nloc * status.MPI_TAG;
                for (j = 0; j < idim; j++) {
                    C[j * jdim + k] = myC[j * counter + i];
                }
            }
        }
        //MPI_Bcast(A, idim*kdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 100, MPI_COMM_WORLD);
        t1 = get_wall_time() - t1;

        //printf("C matrix sample: \n");
        //print_sample(idim, jdim, C, 2, 10);
        // error calculation
        err = error_calc(idim, jdim, actualC, C);

        printf("Finished in %lf seconds\n", t1);
        t1 *= (1.e+09);
        nops = (float)2 * idim * kdim * jdim;
        printf("Performance = %f GFLOPs\n", nops / t1);
        printf("Error: %f\n", err);
    }
    else {
        //printf("I am here too.  From id: %d\n", myid);
        MPI_Bcast(rowval, newdim, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(colval, newdim, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(value, newdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //printf("Proc %d recieved matrix A.\n", myid);
        nloc = (int)(jdim + nprocs - 1) / nprocs;
        if (myid == (nprocs - 1))
            nloc = jdim - nloc * (nprocs - 1);
        //printf("proc %d using nloc of %d.\n", myid, nloc);
        myB = (float*)malloc(nloc * kdim * sizeof(float));
        MPI_Recv(myB, nloc * kdim, MPI_FLOAT, 0, myid, MPI_COMM_WORLD, &status);
        //printf("Proc %d recieved part of B.\n", myid);
        //MPI_Bcast(A, idim*kdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 100, MPI_COMM_WORLD);
        myC = (float*)malloc(idim * nloc * sizeof(float));
        for (i = 0; i < rowlen - 1; i++)
            for (k = rowval[i]; k < rowval[i + 1]; k++)
                for (j = 0; j < nloc; j++)
                    myC[i * nloc + j] += value[k] * myB[colval[k] * nloc + j];
        //matrix_mult(nloc, jdim, kdim, myA, B, myC);
        MPI_Send(myC, nloc * idim, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
        //printf("Proc %d sent part of C.\n", myid);
    }
    /*
        matrix_mult(nloc, jdim, kdim, myA, B, myC);
        if(myid == 0) {
          // error calculation
          err = error_calc(idim, jdim, actualC, C);
          t1 = wctime() - t1;
          printf("Finished in %lf seconds\n", t1);
          t1 *= (1.e+09);
          nops = (float) 2 * idim * kdim * jdim;
          printf("Performance = %f GFLOPs\n", nops/t1);
          printf("Error: %f\n", err);
        } else {
          1;
        }
    */
    MPI_Finalize();
    return(0);
}
