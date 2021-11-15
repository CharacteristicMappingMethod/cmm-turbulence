// this function contains all functions in fourier space and helper functions for dealing with cufftComplex variables
#ifndef __CMM_FFT_H__
#define __CMM_FFT_H__

#include <cufft.h>
#include <cufftXt.h>
#include "../grid/cmm-grid2d.h"

// it's not bad to define them several times, right?
#define PI 		3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
#define twoPI 	6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136

// fourier functions for hermitian data / with D2Z and Z2D
__global__ void k_fft_lap_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid);
__global__ void k_fft_iLap_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid);
__global__ void k_fft_dx_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid);
__global__ void k_fft_dy_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid);

//variable type conversion
void real_to_comp(double *varR, cufftDoubleComplex *varC, long int N);
void comp_to_real(cufftDoubleComplex *varC, double *varR, long int N);
__global__ void k_real_to_comp(double *varR, cufftDoubleComplex *varC, int NX, int NY);
__global__ void k_comp_to_real(cufftDoubleComplex *varC, double *varR, int NX, int NY);

// complex grid scalings
__global__ void k_fft_cut_off_scale_h(cufftDoubleComplex *W, TCudaGrid2D Grid, double freq);
__global__ void k_fft_grid_move(cufftDoubleComplex *In, cufftDoubleComplex *Out, TCudaGrid2D Grid_c, TCudaGrid2D Grid_s);

__global__ void k_normalize_h(cufftDoubleComplex *F, TCudaGrid2D Grid);


#endif
