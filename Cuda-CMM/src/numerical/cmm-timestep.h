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

#ifndef __CMM_TIMESTEP_H__
#define __CMM_TIMESTEP_H__

#include "../grid/cmm-grid2d.h"

// first order
__device__ void euler_exp(double *psi, double *x_in, double *x_out, TCudaGrid2D Grid, double dt);
__device__ void euler_exp_b(double *psi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);

// second order
__device__ void RK2_heun(double *psi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK2_heun_b(double *psi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void adam_bashford_2_pc(double *psi, double *x_in, double *x_out, TCudaGrid2D Grid, double dt);
__device__ void adam_bashford_2_pc_b(double *psi, double *x_in, double *x_out, TCudaGrid2D Grid, double dt);

// third order with third order lagrange interpolation
__device__ void RK3_classical(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK3_classical_b(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK3_optimized(double *psi, double *x_in, double *x_out, TCudaGrid2D Grid, double dt);
__device__ void RK3_optimized_b(double *psi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);

// fourth order with fourth order lagrange interpolation
__device__ void RK4_classical(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK4_classical_b(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK4_optimized(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK4_optimized_b(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);


__host__ __device__ double get_L_coefficient(double *t, double t_next, int loop_ctr, int i_point, int l_order);


#endif
