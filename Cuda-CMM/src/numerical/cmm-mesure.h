#ifndef __CUDA_MESURE_2D_H__
#define __CUDA_MESURE_2D_H__

#include "../grid/cmm-grid2d.h"
#include "../numerical/cmm-fft.h"

#include <math.h>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


void Compute_Energy(double *E, double *psi, TCudaGrid2D Grid);
void Compute_Enstrophy(double *E, double *W, TCudaGrid2D Grid);
void Compute_Enstrophy_fourier(double *E, cufftDoubleComplex *W, TCudaGrid2D Grid);
void Compute_Palinstrophy(TCudaGrid2D Grid, double *Pal, double *W_real, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);


void NDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
void iNDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
__global__ void NDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int NX, int Np);
__global__ void iNDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int N_grid);

#endif


