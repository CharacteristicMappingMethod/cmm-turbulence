#ifndef __CMM_TIMESTEP_H__
#define __CMM_TIMESTEP_H__

#include "../numerical/cmm-hermite.h"

#ifdef __CUDACC__

	// first order
	__device__ void euler_exp(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);

	// second order
	__device__ void RK2_heun(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);
	__device__ void adam_bashford_2_pc(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);

	// third order with third order lagrange interpolation
	__device__ void RK3_classical(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);
	__device__ void RK3_optimized(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);

	// fourth order with fourth order lagrange interpolation
	__device__ void RK4_classical(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);
	__device__ void RK4_optimized(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt);

#endif

#endif
