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

#ifndef __CUDA_HERMITE_2D_H__
#define __CUDA_HERMITE_2D_H__

#include "../grid/cmm-grid2d.h"


//Hermite Basis functions//
// cubic functions with their first derivatives, functions 1 and 2
#define H_f1_3(x) (1 - 3*(x)*(x) + 2*(x)*(x)*(x))
#define H_f1x_3(x) (- 6 + 6*(x)) * (x)
#define H_f2_3(x) ((x)*(1 - (x))*(1 - (x)))
#define H_f2x_3(x) (1 + 3*(x)*(x) - 4*(x))
// cubic functions with their first derivatives, functions 3 and 4, are being used due to less computations
#define H_f3_3(x)  (3 - 2*(x)) * (x)*(x)
#define H_f3x_3(x) (6 - 6*(x)) * (x)
#define H_f4_3(x)  (  (x) - 1) * (x)*(x)
#define H_f4x_3(x) (3*(x) - 2) * (x)

// quintic functions with their first derivatives, functions 4-6 are represented due to being more simple
#define H_f4_5(x)  ( 10 + (-15 +   6*(x)) * (x)) * (x)*(x)*(x)
#define H_f4x_5(x) ( 30 + (-60 +  30*(x)) * (x)) * (x)*(x)
#define H_f5_5(x)  ( -4 + (  7 -   3*(x)) * (x)) * (x)*(x)*(x)
#define H_f5x_5(x) (-12 + ( 28 -  15*(x)) * (x)) * (x)*(x)
#define H_f6_5(x)  (0.5 + ( -1 + 0.5*(x)) * (x)) * (x)*(x)*(x)
#define H_f6x_5(x) (1.5 + ( -4 + 2.5*(x)) * (x)) * (x)*(x)

//CPU functions
//void hermite_interpolation_test();			//test the hermite interpolation functionality



	//the following functions assumes H to be a hermite interpolant on a grid (i*hH, j*hH)
	//and interpolates the value at grid points (i*hF, j*hF) and assigns it to F
	//all interpolation functions are warped
	//a jump is added to diffeo on warping points

//	__global__ void kernel_hermite_interpolation(double *H, double *F, int NXH, int NYH, int NXF, int NYF, double hH, double hF);		//interpolation test

#ifdef __CUDACC__

	__device__ double device_hermite_interpolate_2D   (double *H, double x, double y, TCudaGrid2D Grid);
	__device__ float device_hermite_interpolate_2D   (float *H, float x, float y, TCudaGrid2D Grid);
	__device__ double device_hermite_interpolate_dx_2D(double *H, double x, double y, TCudaGrid2D Grid);
	__device__ float device_hermite_interpolate_dx_2D(float *H, float x, float y, TCudaGrid2D Grid);
	__device__ double device_hermite_interpolate_dy_2D(double *H, double x, double y, TCudaGrid2D Grid);
	__device__ float device_hermite_interpolate_dy_2D(float *H, float x, float y, TCudaGrid2D Grid);

	__device__ void  device_hermite_interpolate_dx_dy_2D(double *H1, double x, double y, double *fx, double *fy, TCudaGrid2D Grid);
	__device__ void  device_hermite_interpolate_dx_dy_2D(float *H1, float x, float y, float *fx, float *fy, TCudaGrid2D Grid);

	__device__ void device_hermite_interpolate_grad_2D(double *H, double *x, double *u, TCudaGrid2D Grid, int n_l);
	__device__ void device_hermite_interpolate_grad_2D(float *H, float *x, float *u, TCudaGrid2D Grid, int n_l);

	__device__ void device_diffeo_interpolate_2D(double *Hx, double *Hy, double x, double y, double *x2,  double *y2, TCudaGrid2D Grid);
	__device__ void device_diffeo_interpolate_2D(float *Hx, float *Hy, float x, float y, float *x2,  float *y2, TCudaGrid2D Grid);
	__device__ double device_diffeo_grad_2D(double *Hx, double *Hy, double x, double y, TCudaGrid2D Grid);
	__device__ float device_diffeo_grad_2D(float *Hx, float *Hy, float x, float y, TCudaGrid2D Grid);

#endif


#endif


