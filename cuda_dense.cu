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

#define TIMES_TO_REPEAT 10

__global__ void matrixMultiply(float* A, float* B, float* C, int I, int J, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;

    if (row < I && col < J) {
        for (int i = 0; i < K; i++) {
            tmp += A[row * K + i] * B[i * J + col];
        }
        C[row * J + col] = tmp;
    }
}

int main(int argc, char* argv[]) {
    int idim = 4000;
    int jdim = 4000;
    int kdim = 4000;
    double t1, times[TIMES_TO_REPEAT];
    float nops, err;
    float* A, * B, * C, * actualC, * Ag, * Bg, * Cg;
    cudaStream_t stream;

    A = (float*)malloc(idim * kdim * sizeof(float));
    B = (float*)malloc(kdim * jdim * sizeof(float));
    C = (float*)malloc(idim * jdim * sizeof(float));
    actualC = (float*)malloc(idim * jdim * sizeof(float));

    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    //printf("A matrix sample: \n");
    //print_sample(idim, kdim, A, 2, 10);
    //printf("B matrix sample: \n");
    //print_sample(kdim, jdim, B, 2, 10);

    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);

    //printf("ActualC matrix sample: \n");
    //print_sample(idim, jdim, actualC, 2, 10);

    cudaMalloc(&Ag, idim * kdim * sizeof(float));
    cudaMalloc(&Bg, kdim * jdim * sizeof(float));
    cudaMalloc(&Cg, idim * jdim * sizeof(float));

    for (int loop_cnt = 0; loop_cnt < TIMES_TO_REPEAT; loop_cnt++) {
        t1 = get_wall_time();
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaMemcpyAsync(Ag, A, idim * kdim * sizeof(float), cudaMemcpyHostToDevice, stream);
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
        //printf("threadsPerBlock: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
        //printf("blocksPerGrid:   (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
        //t1 = wctime();
        matrixMultiply <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (Ag, Bg, Cg, idim, jdim, kdim);
        cudaStreamSynchronize(stream);
        //t1 = wctime() - t1;
        cudaError_t error = cudaGetLastError();
        if (error) {
            printf("CUDA error: %s \n", cudaGetErrorString(error));
            exit(1);
        }
        cudaMemcpyAsync(C, Cg, idim * jdim * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        t1 = get_wall_time() - t1;
        times[loop_cnt] = t1;
        if (loop_cnt != (TIMES_TO_REPEAT - 1))
            zero_init(idim, jdim, C);
    }

    //printf("C matrix sample: \n");
    //print_sample(idim, jdim, C, 2, 10);

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
    cudaFree(Ag);
    cudaFree(Bg);
    cudaFree(Cg);
    cudaStreamDestroy(stream);
    return(0);
}
