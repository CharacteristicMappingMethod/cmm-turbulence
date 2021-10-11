#include "cmm-fft.h"

/*******************************************************************
*						  Fourier operations					   *
*******************************************************************/

// laplacian in fourier space - multiplication by kx**2 and ky**2
__global__ void k_fft_lap(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h)						// Laplace operator in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	if(iX == 0 && iY == 0)
	{
		val_out[0].x = val_out[0].y = 0;
		return;
	}

	int In = iY*NX + iX;

	double kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);						// twoPI/(h*NX) = 1
	double ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);
	double k2 = kx*kx + ky*ky;

	val_out[In].x = -val_in[In].x * k2;
	val_out[In].y = -val_in[In].y * k2;
}


// inverse laplacian in fourier space - division by kx**2 and ky**2
__global__ void k_fft_iLap(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h)						// Inverse laplace operator in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	if(iX == 0 && iY == 0)
	{
		val_out[0].x = val_out[0].y = 0;
		return;
	}

	int In = iY*NX + iX;

	double kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);						// twoPI/(h*NX) = twoPI/(twoPI/NX*NX) = 1
	double ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);
	double k2 = kx*kx + ky*ky;

	val_out[In].x = -val_in[In].x / k2;
	val_out[In].y = -val_in[In].y / k2;
}
// try to setup callback function for functions to get rid of one complex trash variable
//__device__ cufftComplex CB_Input_iLap(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
//{
//	//index
//	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
//	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
//
//	double* helper_array = (double*)callerInfo;
//	int NX = (int)helper_array[0];
//	int NY = (int)helper_array[1];
//	double h = (double)helper_array[2];
//
//	// In is already given by offset
//	// int In = iY*NX + iX;
//
//	double kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);
//	double ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);
//	double k2 = kx*kx + ky*ky;
//
////	if ((int)offset == 100) {
//		printf("In=100, NX=%d, NY=%d, h=%f, k2=%f", NX, NY, h, k2);
////	}
//
//	((cufftComplex*)dataIn)[offset].x = - ((cufftComplex*)dataIn)[offset].x / k2;
//	((cufftComplex*)dataIn)[offset].y = - ((cufftComplex*)dataIn)[offset].y / k2;
//
//    return ((cufftComplex*)dataIn)[offset];
//}


// x derivative in fourier space, multiplication by kx
__global__ void k_fft_dx(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h)						// x derivative in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	int In = iY*NX + iX;

	double kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);

	val_out[In].x = -val_in[In].y * kx;
	val_out[In].y =  val_in[In].x * kx;
}


// y derivative in fourier space, multiplication by ky
__global__ void k_fft_dy(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h)						// y derivative in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	int In = iY*NX + iX;

	double ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);

	val_out[In].x = -val_in[In].y * ky;
	val_out[In].y =  val_in[In].x * ky;
}




/*******************************************************************
*			Transformation functions for cuFFT Complex
*******************************************************************/

// real to complex, initialize imaginary part as 0
__global__ void k_real_to_comp(double *varR, cufftDoubleComplex *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	int In = iY*NX + iX;

	varC[In].x = varR[In];
	varC[In].y = 0.0;
}


// complex to real, cut off imaginary part
__global__ void k_comp_to_real(cufftDoubleComplex *varC, double *varR, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	int In = iY*NX + iX;

	varR[In] = varC[In].x;
}


// real to complex for 4 times the size for hermitian arrays
__global__ void k_real_to_compl_H(double *varR, cufftDoubleComplex *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	// not scattered
	int In = 4*(iY*NX + iX);

	#pragma unroll
	for (int i=0; i<4; i++) {
		varC[In+i].x = varR[In+i];
	}
	varC[In].y = varC[In+1].y = varC[In+2].y = varC[In+3].y = 0.0;
}


// complex t oreal for 4 times the size for hermitian arrays
__global__ void k_compl_to_real_H(cufftDoubleComplex *varC, double *varR, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	// not scattered
	int In = 4*(iY*NX + iX);

	#pragma unroll
	for (int i=0; i<4; i++) varR[In+i] = varC[In+i].x;
}



/*******************************************************************
*						Complex grid scalings
*******************************************************************/

// cut at given frequencies in a round circle
__global__ void k_fft_cut_off_scale(cufftDoubleComplex *W, int NX, double freq)
{
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	int In = iY*NX + iX;

	int i = In/NX;
	int j = In%NX;
	// take symmetry into account
	if (i > NX/2) i = NX-i;
	if (j > NX/2) j = NX-j;
	// cut at frequency in a round circle
	if ((i*i + j*j) > freq*freq || In == 0) {
		W[In].x = 0;
		W[In].y = 0;
	}
}

/*
 * zero padding, all outer elements are moved, conserving symmetry of spectrum
 * take care about entries when making grid larger, data needs to be set to zero first with memset
 * different functions decide between direction of change (kernel onto other grid, or other onto kernel grid)
 * kernel properties set information for c-values
 */
__global__ void k_fft_grid_add(cufftDoubleComplex *In, cufftDoubleComplex *Out, double Nc, double Ns) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = (blockDim.y * blockIdx.y + threadIdx.y);

	int Inc = iYc*Nc + iXc;

	// get new positions
	int iXs = iXc;
	int iYs = iYc;
	// shift upper half to the outer edge
	if (iXc > Nc/2) {
		iXs += Ns - Nc;
	}
	if (iYc > Nc/2) {
		iYs += Ns - Nc;
	}
	// get index in new system
	int Ins = iYs*Ns + iXs;
	// transcribe, add values and leave hole in the middle
	Out[Ins].x = In[Inc].x;
	Out[Ins].y = In[Inc].y;
}
__global__ void k_fft_grid_remove(cufftDoubleComplex *In, cufftDoubleComplex *Out, double Nc, double Ns) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = (blockDim.y * blockIdx.y + threadIdx.y);

	int Inc = iYc*Nc + iXc;

	// get new positions
	int iXs = iXc;
	int iYs = iYc;
	// shift upper half to the outer edge
	if (iXc > Nc/2) {
		iXs += Ns - Nc;
	}
	if (iYc > Nc/2) {
		iYs += Ns - Nc;
	}
	// get index in new system
	int Ins = iYs*Ns + iXs;
	// transcribe, inverse of add so we actually ignore the middle entries
	Out[Inc].x = In[Ins].x;
	Out[Inc].y = In[Ins].y;
}


// divide all values by grid size
__global__ void k_normalize(cufftDoubleComplex *F, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NY)
		return;

	int In = iY*NX + iX;
	double N = NX*NY;

	F[In].x /= (double)N;
	F[In].y /= (double)N;
}
