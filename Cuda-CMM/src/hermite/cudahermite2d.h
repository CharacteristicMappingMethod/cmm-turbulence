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
	
	
	__device__ ptype device_hermite_mult(ptype *H, ptype b[][4], int I[], long int N, ptype h);
		
	__global__ void kernel_hermite_interpolation(ptype *H, ptype *F, int NXH, int NYH, int NXF, int NYF, ptype hH, ptype hF);		//interpolation test
	
	__device__ ptype device_hermite_interpolate   (ptype *H, ptype x, ptype y, int NX, int NY, ptype h);
	__device__ ptype device_hermite_interpolate_dx(ptype *H, ptype x, ptype y, int NX, int NY, ptype h);
	__device__ ptype device_hermite_interpolate_dy(ptype *H, ptype x, ptype y, int NX, int NY, ptype h);
	
	__device__ void  device_hermite_interpolate_dx_dy(ptype *H1, ptype x, ptype y, ptype *fx, ptype *fy, int NX, int NY, ptype h);
	__device__ void  device_hermite_interpolate_dx_dy_1(ptype *H1, ptype x, ptype y, ptype *u1, ptype *v1, int NX, int NY, ptype h);  // make it easier for RKthree, avoid redundant operations
	__device__ void  device_hermite_interpolate_dx_dy_3(ptype *H1, ptype *H2, ptype *H3, ptype x, ptype y, ptype *u1, ptype *v1, ptype *u2, ptype *v2, ptype *u3, ptype *v3, int NX, int NY, ptype h);  // make it easier for RKthree, avoid redundant operations

	__device__ void  device_diffeo_interpolate(ptype *Hx, ptype *Hy, ptype x, ptype y, ptype *x2,  ptype *y2, int NX, int NY, ptype h);	
	__device__ ptype device_diffeo_grad(ptype *Hx, ptype *Hy, ptype x, ptype y, int NX, int NY, ptype h);	
	
	
#endif




#endif


