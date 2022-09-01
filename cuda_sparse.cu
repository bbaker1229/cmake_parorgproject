#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <Common/helper_functions.h>
#include <Common/helper_cuda.h>
extern "C"
{
#include "tools/tools.h"
}

#define TIMES_TO_REPEAT 1

__global__ void sparseMultiply(int* rowvec, int* colvec, float* valvec, float* B, float* C, int I, int J, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;

    if (row < I - 1 && col < J) {
        for (int i = rowvec[row]; i < rowvec[row + 1]; i++) {
            tmp += valvec[i] * B[colvec[i] * J + col];
        }
        C[row * J + col] = tmp;
    }
}

int main(int argc, char* argv[]) {
    int idim = 400;
    int jdim = 400;
    int kdim = 400;
    int rowlen, vallen;
    long int newdim;
    double t1, times[TIMES_TO_REPEAT];
    float nops, per, err;
    float* A, * B, * C, * actualC, * Bg, * Cg, * valg;
    int* rowg, * colg;
    cudaStream_t stream;

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

    cudaMalloc(&rowg, newdim * sizeof(int));
    cudaMalloc(&colg, newdim * sizeof(int));
    cudaMalloc(&valg, newdim * sizeof(float));
    cudaMalloc(&Bg, kdim * jdim * sizeof(float));
    cudaMalloc(&Cg, idim * jdim * sizeof(float));

    for (int loop_cnt = 0; loop_cnt < TIMES_TO_REPEAT; loop_cnt++) {
        printf("cnt: %d\n", loop_cnt);
        t1 = get_wall_time();
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaMemcpyAsync(rowg, rowval, newdim * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(colg, colval, newdim * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(valg, value, newdim * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Bg, B, kdim * jdim * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Cg, C, idim * jdim * sizeof(float), cudaMemcpyHostToDevice, stream);

        dim3 threadsPerBlock(jdim * idim, 2);
        dim3 blocksPerGrid(1, 1);
        if (idim * jdim > 512) {
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 32;
            blocksPerGrid.x = ceil((double)jdim / (double)threadsPerBlock.x);
            blocksPerGrid.y = ceil((double)idim / (double)threadsPerBlock.y);
        }
        cudaStreamSynchronize(stream);
        sparseMultiply <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (rowg, colg, valg, Bg, Cg, rowlen, jdim, kdim);
        cudaStreamSynchronize(stream);

        cudaError_t error = cudaGetLastError();
        if (error) {
            printf("CUDA error: %s \n", cudaGetErrorString(error));
            exit(1);
        }
        cudaMemcpyAsync(C, Cg, idim * jdim * sizeof(float), cudaMemcpyDeviceToHost, stream);
        t1 = get_wall_time() - t1;
        times[loop_cnt] = t1;
        if (loop_cnt != (TIMES_TO_REPEAT - 1))
            zero_init(idim, jdim, C);
    }

    // error calculation
    err = error_calc(idim, jdim, actualC, C);

    t1 = 0.0;
    for (int i = 0; i < TIMES_TO_REPEAT; i++)
        t1 += times[i];
    t1 /= (float) TIMES_TO_REPEAT;
    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float)2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n", nops / t1);
    printf("Error: %f\n", err);
    cudaFree(rowg);
    cudaFree(colg);
    cudaFree(valg);
    cudaFree(Bg);
    cudaFree(Cg);
    cudaStreamDestroy(stream);
    return(0);
}
