#ifndef __CUDA_MESURE_2D_H__
#define __CUDA_MESURE_2D_H__
#include "../grid/cudagrid2d.h"
#include "../simulation/cudasimulation2d.h"






	


#ifdef __CUDACC__
	

__global__ void Compute_Energy(ptype *E, ptype *psi, int N, int NX, int NY, ptype h);

__global__ void Compute_Enstrophy(ptype *Ens, ptype *W_hat, int N, int NX, int NY, ptype h);

void Compute_Palinstrophy(TCudaGrid2D *Grid_coarse, ptype *Pal, cuPtype *W_hat, cuPtype *W_coarse_hat_dx_dy, cuPtype *W_coarse_complex_dx_dy, ptype *W_coarse_real_dx_dy, cufftHandle cufftPlan_coarse);


#endif




#endif
