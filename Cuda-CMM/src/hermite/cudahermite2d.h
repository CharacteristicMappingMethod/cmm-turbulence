#ifndef __CUDA_HERMITE_2D_H__
#define __CUDA_HERMITE_2D_H__
#include "../grid/cudagrid2d.h"




//Hermite Basis functions//
#define Hf(x) (1 - 3*(x)*(x) + 2*(x)*(x)*(x))
#define Hfx(x) (- 6*(x) + 6*(x)*(x))
#define Hg(x) ((x)*(1 - (x))*(1 - (x)))
#define Hgx(x) (1 + 3*(x)*(x) - 4*(x))


//CPU functions
void hermite_interpolation_test();			//test the hermite interpolation functionality




#ifdef __CUDACC__
	
	
	
	//the following functions assumes H to be a hermite interpolant on a grid (i*hH, j*hH)
	//and interpolates the value at grid points (i*hF, j*hF) and assigns it to F
	//all interpolation functions are warped
	//a jump is added to diffeo on warping points
	
	
	__device__ double device_hermite_mult(double *H, double b[][4], int I[], long int N, double h);
	__device__ double device_hermite_mult(double *H, double bX[], double bY[], int I[], long int N, double h);
		
	__global__ void kernel_hermite_interpolation(double *H, double *F, int NXH, int NYH, int NXF, int NYF, double hH, double hF);		//interpolation test
	
	__device__ double device_hermite_interpolate   (double *H, double x, double y, int NX, int NY, double h);
	__device__ double device_hermite_interpolate_dx(double *H, double x, double y, int NX, int NY, double h);
	__device__ double device_hermite_interpolate_dy(double *H, double x, double y, int NX, int NY, double h);
	
	__device__ void  device_hermite_interpolate_dx_dy(double *H1, double x, double y, double *fx, double *fy, int NX, int NY, double h);
	__device__ void  device_hermite_interpolate_dx_dy_1(double *H1, double x, double y, double *u1, double *v1, int NX, int NY, double h);  // make it easier for RKthree, avoid redundant operations
	__device__ void  device_hermite_interpolate_dx_dy_3(double *H1, double *H2, double *H3, double x, double y, double *u1, double *v1, double *u2, double *v2, double *u3, double *v3, int NX, int NY, double h);  // make it easier for RKthree, avoid redundant operations

	__device__ void  device_diffeo_interpolate(double *Hx, double *Hy, double x, double y, double *x2,  double *y2, int NX, int NY, double h);
	__device__ double device_diffeo_grad(double *Hx, double *Hy, double x, double y, int NX, int NY, double h);
	
	
#endif




#endif


