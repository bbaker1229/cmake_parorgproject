#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "tools.h"

void zero_init(int rdim, int cdim, float* A) {
	for (int i = 0; i < rdim; i++) {
		for (int j = 0; j < cdim; j++) {
			A[i * cdim + j] = 0.0;
		}
	}
}

void rand_init(int rdim, int cdim, float* A) {
	for (int i = 0; i < rdim; i++) {
		for (int j = 0; j < cdim; j++) {
			A[i * cdim + j] = (float)rand() / (float)RAND_MAX;
		}
	}
}

void seq_init(int rdim, int cdim, float* A) {
	int count = 0;
	for (int i = 0; i < rdim; i++) {
		for (int j = 0; j < cdim; j++) {
			A[i * cdim + j] = (float)count++;
		}
	}
}

void print_matrix(int rdim, int cdim, float* A) {
	for (int i = 0; i < rdim; i++) {
		printf("[ ");
		for (int j = 0; j < cdim; j++) {
			printf("%1.4f ", A[i * cdim + j]);
		}
		printf("]\n");
	}
}

void matrix_mult(int rdim, int cdim, int kdim, float* A, float* B, float* C) {
	int i, j, k;
	for (i = 0; i < rdim; i++)
		for (k = 0; k < kdim; k++)
			for (j = 0; j < cdim; j++)
				C[i * cdim + j] += A[i * kdim + k] * B[k * cdim + j];
}

float error_calc(int rdim, int cdim, float* A, float* B) {
	int i, j;
	float err = 0.0, t = 0.0;
	for (i = 0; i < rdim; i++) {
		for (j = 0; j < cdim; j++) {
			err += ((A[i * cdim + j] - B[i * cdim + j]) * (A[i * cdim + j] - B[i * cdim + j]));
			t += (A[i * cdim + j] * A[i * cdim + j]);
		}
	}
	return((float)sqrt(err / t));
}

int make_sparse_percent(float per, int rdim, int cdim, float* A) {
	long int maxnums, cnt, check;
	maxnums = (long int)(per * (float)rdim * (float)cdim);
	int* vals;
	vals = (int*)malloc(maxnums * sizeof(int));
	for (long int i = 0; i < maxnums; i++) {
		vals[i] = -1;
	}
	cnt = 0;
	while (cnt < maxnums) {
		long int num = rand() % (rdim * cdim + 1);
		check = 0;
		for (long int i = 0; i < cnt; i++) {
			if (vals[i] == num) {
				check = 1;
				break;
			}
		}
		if (check == 0) {
			vals[cnt] = num;
			cnt++;
		}
	}
	for (int i = 0; i < rdim; i++) {
		for (int j = 0; j < cdim; j++) {
			for (int k = 0; k < cnt; k++) {
				if ((i + 1) * (j + 1) == vals[k]) {
					A[i * cdim + j] = 0.0;
				}
			}
		}
	}
	return rdim * cdim - maxnums + 1;
}

void make_sparse_matrix(int rdim, int cdim, int* rowval, int* colval, float* value, int* rowval_size, int* val_size, float* A) {
	int cnt = 0, cnt1;
	rowval[0] = 0;
	cnt1 = 1;
	for (int i = 0; i < rdim; i++) {
		for (int j = 0; j < cdim; j++) {
			if (A[i * cdim + j] != 0.0) {
				rowval[cnt] = i;
				colval[cnt] = j;
				value[cnt] = A[i * cdim + j];
				cnt++;
			}
		}
		rowval[cnt1] = cnt;
		cnt1++;
	}
	*val_size = cnt;
	*rowval_size = cnt1;
}

//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
				((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else {
		//  Handle error
		return 0;
	}
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time() {
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time() {
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif
