#ifndef __CUDA_GRID_2D_H__
#define __CUDA_GRID_2D_H__

//#include "stdio.h"
//#include "stdlib.h"
//#include "math.h"
//#include "fstream"
//#include "iomanip"
//#include "string.h"
//#include "sstream"
//#include "cstdlib"
//#include <omp.h>
#include <cufft.h>
#include <cufftXt.h>
//#include <iostream>
//#include <string>
//#include <stdio.h>
//#include <time.h>
//#include <curand.h>
//#include <curand_kernel.h>

#define PI 		3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
#define twoPI 	6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136

//#define DISCRET

//the grid is assumed to span the space [0, NX*h] X [0, NY*h]
class TCudaGrid2D
{
	public:
		int NX, NY;
		long int N;
		double h;

		dim3 threadsPerBlock, blocksPerGrid;

		long int sizeNReal, sizeNComplex;

		TCudaGrid2D(int NX, int NY, double xRange);
};

void get_max_min(TCudaGrid2D *G, double *var, double *min, double *max);
void Host_get_max_min(int len, double *Var_min, double *Var_max, double *min, double *max);


#ifdef __CUDACC__

	#define BLOCK_SIZE 16	// most devices have maximum thread size of 1024, i don't know why this is 16*16=256
	#define sm_50  // for Julius, because he has a shitty graphics card

	__global__ void Dev_get_max_min(int len, double *var, double *min, double *max);
#endif

#endif


