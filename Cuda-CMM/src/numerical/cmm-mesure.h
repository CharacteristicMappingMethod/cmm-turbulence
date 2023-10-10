/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/CharacteristicMappingMethod/cmm-turbulence
*
******************************************************************************************************************************/

#ifndef __CUDA_MESURE_2D_H__
#define __CUDA_MESURE_2D_H__

#include "../grid/cmm-grid2d.h"
#include "../numerical/cmm-fft.h"

#include <math.h>

// euler incompressible
void Compute_Energy_H(double &E, CmmVar2D Psi, size_t offset_start=0);
void Compute_Energy(double &E, CmmVar2D Psi, cufftDoubleComplex *Dev_Temp_C1, size_t offset_start=0);
void Compute_Enstrophy(double &E, CmmVar2D Vort, size_t offset_start=0);
void Compute_Enstrophy_fourier(double &E, cufftDoubleComplex *W, TCudaGrid2D Grid);
void Compute_Palinstrophy(double &Pal, CmmVar2D Vort, cufftDoubleComplex *Dev_Temp_C1, size_t offset_start=0);
void Hash_array(char *Hash, double *Dev_var, size_t n);

// vlasov poisson
void Compute_Mass(double &mass, CmmVar2D VarIn, size_t offset_start=0);
void Compute_Momentum(double &P_out, CmmVar2D VarIn, cufftDoubleReal *Dev_Temp_C1, size_t offset_start=0);
void Compute_Total_Energy(double &E_tot, double &E_kin, double &E_pot, CmmVar2D VarKin, CmmVar2D VarPot, cufftDoubleReal *Dev_Temp_C1, size_t offset_start_1=0, size_t offset_start_2=0);
void Compute_Kinetic_Energy(double &E_out, CmmVar2D VarIn, cufftDoubleReal *Dev_Temp_C1, size_t offset_start=0);
void Compute_Potential_Energy(double &E_out, CmmVar2D VarIn, size_t offset_start=0);

// function to compute several global quantities for comparison purposes
// currently: min, max, area weighted avg, area weighted L2 norm
void Compute_min(double &mesure, CmmVar2D Var, size_t offset_start=0);
void Compute_max(double &mesure, CmmVar2D Var, size_t offset_start=0);
void Compute_avg(double &mesure, CmmVar2D Var, size_t offset_start=0);
void Compute_L2(double &mesure, CmmVar2D Var, size_t offset_start=0);
void Compute_min(double &mesure, double* Dev_var, size_t N, size_t offset_start=0);
void Compute_max(double &mesure, double* Dev_var, size_t N, size_t offset_start=0);
void Compute_avg(double &mesure, double* Dev_var, size_t N, size_t offset_start=0);
void Compute_L2(double &mesure, double* Dev_var, size_t N, size_t offset_start=0);


void NDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
void iNDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N);
__global__ void NDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int NX, int Np);
__global__ void iNDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int N_grid);

#endif


