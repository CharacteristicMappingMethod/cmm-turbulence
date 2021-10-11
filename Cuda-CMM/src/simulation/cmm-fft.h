// this function contains all functions in fourier space and helper functions for dealing with cufftComplex variables
#ifndef __FFT_H__
#define __FFT_H__

#include <cufft.h>
#include <cufftXt.h>

#ifdef __CUDACC__

	// it's not bad to define them several times, right?
	#define PI 		3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
	#define twoPI 	6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136

	// fourier functions
	__global__ void k_fft_lap(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);
	__global__ void k_fft_iLap(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);
	__global__ void k_fft_dx(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);
	__global__ void k_fft_dy(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);

	//	__device__ cufftComplex CB_Input_iLap(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

	//variable type conversion
	__global__ void k_real_to_comp(double *varR, cufftDoubleComplex *varC, int NX, int NY);
	__global__ void k_comp_to_real(cufftDoubleComplex *varC, double *varR, int NX, int NY);
	__global__ void k_real_to_comp_H(double *varR, cufftDoubleComplex *varC, int NX, int NY);
	__global__ void k_comp_to_real_H(cufftDoubleComplex *varC, double *varR, int NX, int NY);

	// complex grid scalings
	__global__ void k_fft_cut_off_scale(cufftDoubleComplex *W, int NX, double freq);
	__global__ void k_fft_grid_add(cufftDoubleComplex *In, cufftDoubleComplex *Out, double Nc, double Ns);
	__global__ void k_fft_grid_remove(cufftDoubleComplex *In, cufftDoubleComplex *Out, double Nc, double Ns);

	__global__ void k_normalize(cufftDoubleComplex *F, int NX, int NY);

#endif

#endif
