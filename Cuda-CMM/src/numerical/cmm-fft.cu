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

#include "cmm-fft.h"

#include "../grid/cmm-grid2d.h"

#include "stdio.h"

/*******************************************************************
*						  Fourier operations					   *
*			All oparate on hermitian input for D2Z and Z2D
*			that means, that NX row is (NX/2.0+1)
*******************************************************************/

// functions for hermitian input / for D2Z or Z2D
// laplacian in fourier space - multiplication by kx**2 and ky**2
__global__ void k_fft_lap_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX_fft || iY >= Grid.NY)
		return;

	if(iX == 0 && iY == 0)
	{
		val_out[0].x = val_out[0].y = 0;
		return;
	}

	int In = iY*Grid.NX_fft + iX;

	double kx = twoPI/(Grid.hx*Grid.NX) * (iX - (iX>Grid.NX/2)*Grid.NX);						// twoPI/(h*NX) = 1
	double ky = twoPI/(Grid.hy*Grid.NY) * (iY - (iY>Grid.NY/2)*Grid.NY);
	double k2 = kx*kx + ky*ky;

	val_out[In].x = -val_in[In].x * k2;
	val_out[In].y = -val_in[In].y * k2;
}


// inverse laplacian in fourier space - division by kx**2 and ky**2
__global__ void k_fft_iLap_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX_fft || iY >= Grid.NY)
		return;

	if(iX == 0 && iY == 0)
	{
		val_out[0].x = val_out[0].y = 0;
		return;
	}

	int In = iY*Grid.NX_fft + iX;

	double kx = twoPI/(Grid.hx*Grid.NX) * (iX - (iX>Grid.NX/2)*Grid.NX);  // twoPI/(h*NX) = twoPI/(twoPI/NX*NX) = 1
	double ky = twoPI/(Grid.hy*Grid.NY) * (iY - (iY>Grid.NY/2)*Grid.NY);
	double k2 = kx*kx + ky*ky;

	val_out[In].x = -val_in[In].x / k2;
	val_out[In].y = -val_in[In].y / k2;
}

// inverse laplacian in fourier space - division by kx**2
__global__ void k_fft_iLap_h_1D(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	if(iX >= Grid.NX_fft)
		return;

	if(iX == 0)
	{
		val_out[0].x = val_out[0].y = 0.0;
		return;
	}

	int In = iX;

	double kx = twoPI/(Grid.hx*Grid.NX) * (iX - (iX>Grid.NX/2)*Grid.NX);
	double k2 = kx*kx;

    val_out[In].x = - val_in[In].x /k2;
    val_out[In].y = - val_in[In].y /k2;
}



// 1D x derivative in fourier space, multiplication by kx
__global__ void k_fft_dx_h_1D(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	
	if(iX >= Grid.NX_fft)
		return;

	double kx = twoPI/(Grid.hx*Grid.NX) * (iX - (iX>Grid.NX/2)*Grid.NX);

	double2 temp = val_in[iX];
//	temp.x = val_in[In].x; temp.y = val_in[In].y;

	val_out[iX].x = -temp.y * kx;
	val_out[iX].y =  temp.x * kx;
}

// x derivative in fourier space, multiplication by kx
__global__ void k_fft_dx_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX_fft || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX_fft + iX;

	double kx = twoPI/(Grid.hx*Grid.NX) * (iX - (iX>Grid.NX/2)*Grid.NX);

	double2 temp = val_in[In];
//	temp.x = val_in[In].x; temp.y = val_in[In].y;

	val_out[In].x = -temp.y * kx;
	val_out[In].y =  temp.x * kx;
}


// y derivative in fourier space, multiplication by ky
__global__ void k_fft_dy_h(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX_fft || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX_fft + iX;

	double ky = twoPI/(Grid.hy*Grid.NY) * (iY - (iY>Grid.NY/2)*Grid.NY);

	double2 temp = val_in[In];
//	temp.x = val_in[In].x; temp.y = val_in[In].y;

	val_out[In].x = -temp.y * ky;
	val_out[In].y =  temp.x * ky;
}




/*******************************************************************
*			Transformation functions for cuFFT Complex
*******************************************************************/

/*
 *  real to complex and complex to real functions utilizing strided memory copying
 *  complex values are stored in the form [(x1, y1), (x2, y2), ... (xn, yn)]
 *  real values are stored in the form [ (x1), (x2), ... (xn)]
 *  this basically means, we have to transform data between C[N,2] and R[N,1] 2d-matrices
 */
void real_to_comp(double *varR, cufftDoubleComplex *varC, long int N)
{
	// due to memset, this function is indeed quite slow
	cudaMemcpy2D(varC, sizeof(cufftDoubleComplex), varR, sizeof(double), sizeof(double), N, cudaMemcpyDeviceToDevice);
	// imaginary values have to be set to zero, double casting to offset index by one
	cudaMemset2D((double*)varC + 1, sizeof(cufftDoubleComplex), 0, sizeof(double), N);
}
void comp_to_real(cufftDoubleComplex *varC, double *varR, long int N)
{
	cudaMemcpy2D(varR, sizeof(double), varC, sizeof(cufftDoubleComplex), sizeof(double), N, cudaMemcpyDeviceToDevice);
}


// function to shift all values, here memcpy cannot be used
__global__ void k_copy_shift(double *varR, TCudaGrid2D Grid, int shift, long int length)
{
	//index, assumed 1D, since we want to shift with ascending In
	long int In = (blockDim.x * blockIdx.x + threadIdx.x);

	if (shift > 0) In *= -1;

	if(In >= length or In < 0)
		return;

	// shift values
	varR[In+shift] = varR[In];
}


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

/*******************************************************************
*						Complex grid scalings
*******************************************************************/

__global__ void k_fft_cut_off_scale_h(cufftDoubleComplex *W, TCudaGrid2D Grid, double freq)
{
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= (int)(Grid.NX/2.0+1) || iY >= Grid.NY)
		return;

	int In = iY*(int)(Grid.NX/2.0+1) + iX;

	// take symmetry into account, warp to lower half
	iX += (iX > Grid.NX/2.0) * (Grid.NX - 2*iX);
	iY += (iY > Grid.NY/2.0) * (Grid.NY - 2*iY);
	// cut at frequency in a round circle
	if ((iX*iX + iY*iY) > freq*freq || In == 0) {
//	if (i > freq || j > freq || In == 0) {
		W[In].x = 0;
		W[In].y = 0;
	}
}


/*
 * Grid moving - the new, better zeropad-technology
 * transcribe all elements and add zero for unwanted ones
 * this does not work inline (input=output)!
 */
__global__ void k_fft_grid_move(cufftDoubleComplex *In, cufftDoubleComplex *Out, TCudaGrid2D Grid_out, TCudaGrid2D Grid_in) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iXc >= Grid_out.NX_fft || iYc >= Grid_out.NY || iXc < 0 || iYc < 0)
		return;

	int Inc = iYc*Grid_out.NX_fft + iXc;

	// check if we have to set values to 0, transform index to lower half and compare to gridsize
	if ((iXc + (iXc > Grid_out.NX/2.0) * (Grid_out.NX - 2*iXc) > Grid_in.NX/2.0) || (iYc + (iYc > Grid_out.NY/2.0) * (Grid_out.NY - 2*iYc) > Grid_in.NY/2.0)) {
		Out[Inc].x = Out[Inc].y = 0;
	}
	else {
		// get new positions and shift upper half to the outer edge
		int iXs = iXc + (iXc > Grid_out.NX/2.0) * (Grid_in.NX - Grid_out.NX);
		int iYs = iYc + (iYc > Grid_out.NY/2.0) * (Grid_in.NY - Grid_out.NY);
		// get index in new system
		int Ins = iYs*Grid_in.NX_fft+ iXs;

		// transcribe, add values and leave hole in the middle
		double2 temp = In[Ins];
		Out[Inc].x = temp.x;
		Out[Inc].y = temp.y;

//		if (blockIdx.x+blockIdx.y == 0) {
//			printf("In - %d \t iXc - %d \t iYc - %d \t iXs - %d \t iYs - %d \n", Inc, iXc, iYc, iXs, iYs);
//		}
	}
}

// divide all values by grid size
__global__ void k_normalize_h(cufftDoubleComplex *F, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= (int)(Grid.NX/2.0+1) || iY >= Grid.NY)
		return;

	int In = iY*(int)(Grid.NX/2.0+1) + iX;

	F[In].x /= (double)Grid.N;
	F[In].y /= (double)Grid.N;
}


// divide all values by grid size
__global__ void k_normalize_1D_h(cufftDoubleComplex *F, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);

	int In = iX;

	F[In].x /= (double)Grid.NX;
	F[In].y /= (double)Grid.NX;
}


__global__ void zero_padding_1D(cufftDoubleComplex *In, cufftDoubleComplex *Out, TCudaGrid2D Grid_out, TCudaGrid2D Grid_in) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = 0;

	if(iXc >= Grid_out.NX_fft || iXc < 0 )
		return;

	int Inc = iXc;

	// check if we have to set values to 0, transform index to lower half and compare to gridsize
	if ((iXc + (iXc > Grid_out.NX/2.0) * (Grid_out.NX - 2*iXc) > Grid_in.NX/2.0)) {
		Out[Inc].x = Out[Inc].y = 0;
	}
	else {
		// get new positions and shift upper half to the outer edge
		int iXs = iXc + (iXc > Grid_out.NX/2.0) * (Grid_in.NX - Grid_out.NX);
		// get index in new system
		int Ins = iXs;

		// transcribe, add values and leave hole in the middle
		double2 temp = In[Ins];
		Out[Inc].x = temp.x;
		Out[Inc].y = temp.y;

//		if (blockIdx.x+blockIdx.y == 0) {
//			printf("In - %d \t iXc - %d \t iYc - %d \t iXs - %d \t iYs - %d \n", Inc, iXc, iYc, iXs, iYs);
//		}
	}
}

/*******************************************************************
*								Test NDFT						   *
*******************************************************************/

/*
printf("NDFT\n");

int Np_particles = 16384;
int iNDFT_block, iNDFT_thread = 256;
iNDFT_block = Np_particles/256;
int *f_k, *Dev_f_k;
double *x_1_n, *x_2_n, *p_n, *Dev_x_1_n, *Dev_x_2_n, *Dev_p_n, *X_k_bis;
cufftDoubleComplex *X_k, *Dev_X_k, *Dev_X_k_derivative;

x_1_n = new double[Np_particles];
x_2_n = new double[Np_particles];
p_n = new double[2*Np_particles];
f_k = new int[Grid_NDFT.NX];
X_k = new cufftDoubleComplex[Grid_NDFT.N];
X_k_bis = new double[2*Grid_NDFT.N];
cudaMalloc((void**)&Dev_x_1_n, sizeof(double)*Np_particles);
cudaMalloc((void**)&Dev_x_2_n, sizeof(double)*Np_particles);
cudaMalloc((void**)&Dev_p_n, sizeof(double)*2*Np_particles);
cudaMalloc((void**)&Dev_f_k, sizeof(int)*Np_particles);
cudaMalloc((void**)&Dev_X_k, Grid_NDFT.sizeNComplex);
cudaMalloc((void**)&Dev_X_k_derivative, Grid_NDFT.sizeNComplex);

readRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x1.data");
readRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x2.data");
readRealToBinaryAnyFile(2*Np_particles, p_n, "src/Initial_W_discret/p.data");

for(int i = 0; i < Grid_NDFT.NX; i+=1)
	f_k[i] = i;

cudaMemcpy(Dev_x_1_n, x_1_n, sizeof(double)*Np_particles, cudaMemcpyHostToDevice);
cudaMemcpy(Dev_x_2_n, x_2_n, sizeof(double)*Np_particles, cudaMemcpyHostToDevice);
cudaMemcpy(Dev_p_n, p_n, sizeof(double)*2*Np_particles, cudaMemcpyHostToDevice);
cudaMemcpy(Dev_f_k, f_k, sizeof(int)*Grid_NDFT.NX, cudaMemcpyHostToDevice);

printf("NDFT v_x\n");
NDFT_2D<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX, Np_particles);
printf("iNDFT v_x\n");
iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
cudaMemcpy(x_1_n, Dev_x_1_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
writeRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x_1_ifft.data");

printf("k_fft_dx\n");
k_fft_dx<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
printf("iNDFT v_x/dx\n");
iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k_derivative, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
cudaMemcpy(x_1_n, Dev_x_1_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
writeRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x_1_dx_ifft.data");

cudaMemcpy(X_k, Dev_X_k_derivative, Grid_NDFT.sizeNComplex, cudaMemcpyDeviceToHost);
printf("%lf %lf %lf\n", X_k[0].x, X_k[1].x, X_k[Grid_NDFT.N-1].x);
//writeRealToBinaryAnyFile(2*Np_particles, X_k, "src/Initial_W_discret/X_k.data");

for(int i = 0; i < Grid_NDFT.N; i+=1){
	X_k_bis[2*i] 	= 	X_k[i].x;
	X_k_bis[2*i+1] 	= 	X_k[i].y;
}
writeRealToBinaryAnyFile(2*Grid_NDFT.N, X_k_bis, "src/Initial_W_discret/X_k.data");


printf("NDFT v_y\n");
NDFT_2D<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX, Np_particles);
printf("iNDFT v_y\n");
iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
cudaMemcpy(x_2_n, Dev_x_2_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
writeRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x_2_ifft.data");

printf("k_fft_dy\n");
k_fft_dy<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
printf("iNDFT v_y/dy\n");
iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k_derivative, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
cudaMemcpy(x_2_n, Dev_x_2_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
writeRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x_2_dy_ifft.data");

printf("Fini NDFT\n");
*/
