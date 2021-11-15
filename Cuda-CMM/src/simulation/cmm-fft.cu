#include "cmm-fft.h"

#include "../grid/cudagrid2d.h"

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

	if(iX >= NX || iY >= NX)
		return;

	int In = iY*NX + iX;

	// take symmetry into account
	if (iX > NX/2) iX = NX-iX;
	if (iY > NX/2) iY = NX-iY;
	// cut at frequency in a round circle
	if ((iX*iX + iY*iY) > freq*freq || In == 0) {
//	if (i > freq || j > freq || In == 0) {
		W[In].x = 0;
		W[In].y = 0;
	}
}

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
 * start from the back to ensure, that we do not overwrite values
 * this works inline (input=output) or with two seperate values
 */
__global__ void k_fft_grid_move(cufftDoubleComplex *In, cufftDoubleComplex *Out, TCudaGrid2D Grid_out, TCudaGrid2D Grid_in) {
	int iXc = Grid_out.NX_fft - (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = Grid_out.NY     - (blockDim.y * blockIdx.y + threadIdx.y);

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
		Out[Inc].x = In[Ins].x;
		Out[Inc].y = In[Ins].y;
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

	if(iXc >= Nc || iYc >= Nc)
		return;

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

	if(iXc >= Nc || iYc >= Nc)
		return;

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

__global__ void k_fft_grid_add_h(cufftDoubleComplex *In, cufftDoubleComplex *Out, TCudaGrid2D Grid_c, TCudaGrid2D Grid_s) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iXc >= (int)(Grid_c.NX/2.0+1) || iYc >= Grid_c.NY)
		return;

	int Inc = iYc*(int)(Grid_c.NX/2.0+1) + iXc;

	// get new positions and shift upper half to the outer edge
	int iXs = iXc + (iXc > Grid_c.NX/2.0) * (Grid_s.NX - Grid_c.NX);
	int iYs = iYc + (iYc > Grid_c.NY/2.0) * (Grid_s.NY - Grid_c.NY);
	// get index in new system
	int Ins = iYs*(int)(Grid_s.NX/2.0+1) + iXs;
	// transcribe, add values and leave hole in the middle
	Out[Ins].x = In[Inc].x;
	Out[Ins].y = In[Inc].y;
}
__global__ void k_fft_grid_remove_h(cufftDoubleComplex *In, cufftDoubleComplex *Out, TCudaGrid2D Grid_c, TCudaGrid2D Grid_s) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iXc >= (int)(Grid_c.NX/2.0+1) || iYc >= Grid_c.NY)
		return;

	int Inc = iYc*(int)(Grid_c.NX/2.0+1) + iXc;

	// get new positions and shift upper half to the outer edge
	int iXs = iXc + (iXc > Grid_c.NX/2.0) * (Grid_s.NX - Grid_c.NX);
	int iYs = iYc + (iYc > Grid_c.NY/2.0) * (Grid_s.NY - Grid_c.NY);
	// get index in new system
	int Ins = iYs*(int)(Grid_s.NX/2.0+1) + iXs;
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
