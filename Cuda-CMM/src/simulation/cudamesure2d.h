#ifndef __CUDA_MESURE_2D_H__
#define __CUDA_MESURE_2D_H__
#include "../grid/cudagrid2d.h"
#include "../simulation/cudasimulation2d.h"


//Lagrange polynomial with stencil
//#define L1(t, tp, tm, tmm) ((((t)-(tm))*((t)-(tmm)))/(((tp)-(tm))/((tp)-(tmm))))
//#define L2(t, tp, tm, tmm) ((t-tp)*(t-tmm)/((tm-tp)*(tm-tmm)))
//#define L3(t, tp, tm, tmm) ((t-tp)*(t-tm)/((tmm-tp)*(tmm-tm)))

#ifdef __CUDACC__


__global__ void Compute_Energy(ptype *E, ptype *psi, int N, int NX, int NY, ptype h);

__global__ void Compute_Enstrophy(ptype *Ens, ptype *W, int N, int NX, int NY, ptype h);

void Compute_Palinstrophy(TCudaGrid2D *Grid_coarse, ptype *Pal, ptype *W_real, cuPtype *Dev_Complex, cuPtype *Dev_Hat, cuPtype *Dev_Hat_bis, cufftHandle cufftPlan_coarse);


void NDFT_1D(cuPtype *X_k, ptype *x_n, ptype *p_n, ptype *f_k, int N);

void iNDFT_1D(cuPtype *X_k, ptype *x_n, ptype *p_n, ptype *f_k, int N);

__global__ void NDFT_2D(cuPtype *X_k, ptype *x_n, ptype *p_n, int *f_k, int NX, int Np);

__global__ void iNDFT_2D(cuPtype *X_k, ptype *x_n, ptype *p_n, int *f_k, int N_grid);

//void Lagrange_coef(ptype *Dev_Lagr_coef, ptype t, ptype tp, ptype tm, ptype tmm);

//__device__ void device_Lagrange_inter(ptype *Dev_Psi_real_previous_p_p, ptype *Dev_Psi_previous_p, ptype *Dev_Psi_previous, ptype *Dev_Psi_real, ptype *Dev_Lag_coef, int N);

void Laplacian_vort(TCudaGrid2D *Grid_fine, ptype *Dev_W_fine, cuPtype *Dev_Complex_fine, cuPtype *Dev_Hat_fine, ptype *Dev_lap_fine_real, cuPtype *Dev_lap_fine_complex, cuPtype *Dev_lap_fine_hat, cufftHandle cufftPlan_fine);

__host__ __device__ ptype L1(ptype t, ptype tp, ptype tm, ptype tmm);
__host__ __device__ ptype L2(ptype t, ptype tp, ptype tm, ptype tmm);
__host__ __device__ ptype L3(ptype t, ptype tp, ptype tm, ptype tmm);

#endif




#endif


