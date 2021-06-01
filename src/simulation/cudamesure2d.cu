#include "cudamesure2d.h"
#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"





__global__ void Compute_Energy(ptype *E, ptype *psi, int N, int NX, int NY, ptype h){

	int idx = threadIdx.x;
	int stride_x = blockDim.x;

	for(int i = idx; i < N; i+=stride_x){
		// atomicAdd(E, 0.5*h * h * (psi[i + N] * psi[i + N] + psi[i + 2 * N] * psi[i + 2 * N]));
	}

}


__global__ void Compute_Enstrophy(ptype *Ens, ptype *W, int N, int NX, int NY, ptype h){

	int idx = threadIdx.x;
	int stride_x = blockDim.x;

	for(int i = idx; i < N; i+=stride_x){
		// atomicAdd(Ens, 0.5 * h * h * (W[i] * W[i]) );
	}
  
}


void Compute_Palinstrophy(TCudaGrid2D *Grid_coarse, ptype *Pal, cuPtype *W_hat, cuPtype *W_coarse_hat_dx_dy, cuPtype *W_coarse_complex_dx_dy, ptype *W_coarse_real_dx_dy, cufftHandle cufftPlan_coarse){

	kernel_fft_dx<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_hat, W_coarse_hat_dx_dy, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);												// x-derivative in Fourier space
	kernel_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_hat, &W_coarse_hat_dx_dy[Grid_coarse->N], Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);								// y-derivative in Fourier space
  
  	cufftExecZ2Z(cufftPlan_coarse, W_coarse_hat_dx_dy, W_coarse_complex_dx_dy, CUFFT_INVERSE);																												// stream function from vorticity on coarse grid
	cufftExecZ2Z(cufftPlan_coarse, &W_coarse_hat_dx_dy[Grid_coarse->N], &W_coarse_complex_dx_dy[Grid_coarse->N], CUFFT_INVERSE);																			// x-derivative of the stream function = v-velocity 
	
	kernel_complex_to_real<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_coarse_real_dx_dy, W_coarse_complex_dx_dy, Grid_coarse->NX, Grid_coarse->NY);
	/*
	for(int i = idx; i < Grid_coarse->N; i+=stride_x){
		// atomicAdd(Pal, Grid_coarse->h * Grid_coarse->h * (1) ); //use  into the Fourier space WIP
	}
	*/
	
}


// kernel_fft_dx<<<x,x>>>() remplacer psi par omega
// meme chose _dy

// cufftExecZ2Z(cufftPlan_coarse, Psi_hat, Psi_complex, CUFFT_INVERSE)


// kernel_fft_dx<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Psi_hat, &Psi_hat[Grid_coarse->N], Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);										// x-derivative in Fourier space
// kernel_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Psi_hat, &Psi_hat[2*Grid_coarse->N], Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);										// y-derivative in Fourier space				



















	
