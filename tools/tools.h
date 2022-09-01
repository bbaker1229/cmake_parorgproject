#pragma once
#ifndef SRC_TOOLS_H_
#define SRC_TOOLS_H_

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

double get_wall_time();

double get_cpu_time();

void print_matrix(int rdim, int cdim, float* A);

void zero_init(int rdim, int cdim, float* A);

void rand_init(int rdim, int cdim, float* A);

int make_sparse_percent(float per, int rdim, int cdim, float* A);

void make_sparse_matrix(int rdim, int cdim, int* rowval, int* colval, float* value, int* rowval_size, int* val_size, float* A);

void seq_init(int rdim, int cdim, float* A);

void matrix_mult(int rdim, int cdim, int kdim, float* A, float* B, float* C);

float error_calc(int rdim, int cdim, float* A, float* B);

#endif