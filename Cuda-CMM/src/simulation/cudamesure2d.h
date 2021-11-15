#ifndef __CUDA_MESURE_2D_H__
#define __CUDA_MESURE_2D_H__

#include "../grid/cudagrid2d.h"
#include "../simulation/cmm-fft.h"
#include "../grid/cudagrid2d.h"

#include <math.h>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


//Lagrange polynomial with stencil
//#define L1(t, tp, tm, tmm) ((((t)-(tm))*((t)-(tmm)))/(((tp)-(tm))/((tp)-(tmm))))
//#define L2(t, tp, tm, tmm) ((t-tp)*(t-tmm)/((tm-tp)*(tm-tmm)))
//#define L3(t, tp, tm, tmm) ((t-tp)*(t-tm)/((tmm-tp)*(tmm-tm)))

void Compute_Energy(double *E, double *psi, TCudaGrid2D Grid);
void Compute_Enstrophy(double *E, double *W, TCudaGrid2D Grid);
void Compute_Palinstrophy(TCudaGrid2D Grid, double *Pal, double *W_real, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);


void NDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
void iNDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
__global__ void NDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int NX, int Np);
__global__ void iNDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int N_grid);

//void Lagrange_coef(double *Dev_Lagr_coef, double t, double tp, double tm, double tmm);

//__device__ void device_Lagrange_inter(double *Dev_Psi_real_previous_p_p, double *Dev_Psi_previous_p, double *Dev_Psi_previous, double *Dev_Psi_real, double *Dev_Lag_coef, int N);

//void Laplacian_vort(TCudaGrid2D Grid_fine, double *Dev_W_fine, cufftDoubleComplex *Dev_Complex_fine, cufftDoubleComplex *Dev_Hat_fine, double *Dev_lap_fine_real, cufftDoubleComplex *Dev_lap_fine_complex, cufftDoubleComplex *Dev_lap_fine_hat, cufftHandle cufftPlan_fine);

#endif


