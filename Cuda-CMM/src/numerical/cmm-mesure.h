/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
******************************************************************************************************************************/

#ifndef __CUDA_MESURE_2D_H__
#define __CUDA_MESURE_2D_H__

#include "../grid/cmm-grid2d.h"
#include "../numerical/cmm-fft.h"

#include <math.h>


void Compute_Energy(double &E, double *psi, TCudaGrid2D Grid);
void Compute_Enstrophy(double &E, double *W, TCudaGrid2D Grid);
void Compute_Enstrophy_fourier(double &E, cufftDoubleComplex *W, TCudaGrid2D Grid);
void Compute_Palinstrophy(TCudaGrid2D Grid, double &Pal, double *W_real, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);

void Hash_array(TCudaGrid2D Grid, char *Hash, double *dev_array);

void NDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
void iNDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
__global__ void NDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int NX, int Np);
__global__ void iNDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int N_grid);

#endif


