#ifndef __CMM_TIMESTEP_H__
#define __CMM_TIMESTEP_H__

#include "../grid/cmm-grid2d.h"

// first order
__device__ void euler_exp(double *psi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);

// second order
__device__ void RK2_heun(double *psi, double *d_L1,
		double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void adam_bashford_2_pc(double *psi, double *x_in, double *x_out, TCudaGrid2D Grid, double dt);

// third order with third order lagrange interpolation
__device__ void RK3_classical(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK3_optimized(double *psi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);

// fourth order with fourth order lagrange interpolation
__device__ void RK4_classical(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);
__device__ void RK4_optimized(double *psi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order);


__host__ __device__ double get_L_coefficient(double *t, double t_next, int loop_ctr, int i_point, int l_order);


#endif
