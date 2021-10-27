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
//	if (i > freq || j > freq || In == 0) {
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
