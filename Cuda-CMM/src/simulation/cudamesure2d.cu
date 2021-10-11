#include "cudamesure2d.h"
#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"




__global__ void Compute_Energy(double *E, double *psi, int N, int NX, int NY, double h){


    int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NX)
		return;

	int In = iY*NX + iX;

	// atomicAdd defined for double at sm_60
	#ifndef sm_50
    	atomicAdd(E, 0.5*h * h * (psi[In + N] * psi[In + N] + psi[In + 2 * N] * psi[In + 2 * N]));
	#else
    	*E += 0.5*h * h * (psi[In + N] * psi[In + N] + psi[In + 2 * N] * psi[In + 2 * N]);
	#endif
/*	int idx = threadIdx.x;
	int stride_x = blockDim.x;

	for(int i = idx; i < N; i+=stride_x){
		atomicAdd(E, 0.5*h * h * (psi[i + N] * psi[i + N] + psi[i + 2 * N] * psi[i + 2 * N]));
	}
    */
}


__global__ void Compute_Enstrophy(double *Ens, double *W, int N, int NX, int NY, double h){

    int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NX)
		return;

	int In = iY*NX + iX;

	// atomicAdd defined for double at sm_60
	#ifndef sm_50
    	atomicAdd(Ens, 0.5 * h * h * (W[In] * W[In]));
	#else
    	*Ens += 0.5 * h * h * (W[In] * W[In]);
	#endif
/*	int idx = threadIdx.x;
	int stride_x = blockDim.x;

	for(int i = idx; i < N; i+=stride_x){
		atomicAdd(Ens, 0.5 * h * h * (W[i] * W[i]));
	}
    */
}


// two functions for host, because julius' computer doesn't support atomics for doubles
void Compute_Energy_Host(double *E, double *psi, int N, double h){
	for(int i = 0; i < N; i+=1){
    	*E += 0.5*h * h * (psi[i + N] * psi[i + N] + psi[i + 2 * N] * psi[i + 2 * N]);
	}
}
void Compute_Enstrophy_Host(double *Ens, double *W, int N, double h){
	for(int i = 0; i < N; i+=1){
		*Ens += 0.5 * h * h * (W[i] * W[i]);
	}
}


// attention: Palinstrophy does not work for now with only two trash variables
// i have to rewrite this in different forms
void Compute_Palinstrophy(TCudaGrid2D *Grid_coarse, double *Pal, double *W_real, cufftDoubleComplex *Dev_Complex, cufftDoubleComplex *Dev_Hat, cufftDoubleComplex *Dev_Hat_bis, cufftHandle cufftPlan_coarse){

	double *Host_W_coarse_real_dx_dy = new double[2*Grid_coarse->N];
	
	k_real_to_comp<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Hat, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat, Dev_Complex, CUFFT_FORWARD);	
	k_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Complex, Grid_coarse->NX, Grid_coarse->NY);
	
	k_fft_dx<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Complex, Dev_Hat, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h); 					// x-derivative in Fourier space
	k_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Complex, Dev_Hat_bis, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h); 				// y-derivative in Fourier space
	
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat, Dev_Complex, CUFFT_INVERSE);
    cudaDeviceSynchronize();
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_bis, Dev_Hat, CUFFT_INVERSE);
	
	cudaMemcpy(Host_W_coarse_real_dx_dy, (cufftDoubleReal*)Dev_Complex, Grid_coarse->N, cudaMemcpyDeviceToHost);
	cudaMemcpy(&Host_W_coarse_real_dx_dy[Grid_coarse->N], (cufftDoubleReal*)Dev_Hat, Grid_coarse->N, cudaMemcpyDeviceToHost);


    for(int i = 0; i < Grid_coarse->N; i+=1){
		*Pal += (Grid_coarse->h) * (Grid_coarse->h) * (Host_W_coarse_real_dx_dy[i] * Host_W_coarse_real_dx_dy[i] + Host_W_coarse_real_dx_dy[i + Grid_coarse->N] * Host_W_coarse_real_dx_dy[i + Grid_coarse->N]);
	}

	*Pal = 0.5*(*Pal);
	
}


// compute palinstrophy using fourier transformations - a bit expensive but ca marche
void Compute_Palinstrophy_fourier(TCudaGrid2D *Grid_coarse, double *Pal, double *W_real, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, cufftHandle cufftPlan_coarse){

	double *Host_W_coarse_real_dx_dy = new double[2*Grid_coarse->N];

	// round 1: dx dervative
	k_real_to_comp<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Temp_C1, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Grid_coarse->NX, Grid_coarse->NY);
	k_fft_dx<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h); 					// x-derivative in Fourier space
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);
	cudaMemcpy(Host_W_coarse_real_dx_dy, (cufftDoubleReal*)Dev_Temp_C2, Grid_coarse->N, cudaMemcpyDeviceToHost);

	// round 1: dy dervative
	k_real_to_comp<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Temp_C1, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Grid_coarse->NX, Grid_coarse->NY);
	k_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h); 					// y-derivative in Fourier space
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);
	cudaMemcpy(&Host_W_coarse_real_dx_dy[Grid_coarse->N], (cufftDoubleReal*)Dev_Temp_C2, Grid_coarse->N, cudaMemcpyDeviceToHost);

	// now compute actual palinstrophy and add everything together
    for(int i = 0; i < Grid_coarse->N; i+=1){
		*Pal += 0.5 * (Grid_coarse->h) * (Grid_coarse->h) * (Host_W_coarse_real_dx_dy[i] * Host_W_coarse_real_dx_dy[i] + Host_W_coarse_real_dx_dy[i + Grid_coarse->N] * Host_W_coarse_real_dx_dy[i + Grid_coarse->N]);
	}
}


//// compute palinstrophy using hermite array - cheap, but we need the vorticity hermite for that, only works for fine array, can be kernelized too actually, but not now
//void Compute_Palinstrophy_hermite(TCudaGrid2D *Grid_fine, double *Pal, double *W_H_real){
//
//	// dx and dy values are in position 2/4 and 3/4, copy to host
//	double *Host_W_coarse_real_dx_dy = new double[2*Grid_fine->N];
//	cudaMemcpy(Host_W_coarse_real_dx_dy, &W_H_real[Grid_fine->N], 2*Grid_fine->N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(Host_W_coarse_real_dx_dy, &W_H_real[Grid_fine->N], 2*Grid_fine->N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(Host_W_coarse_real_dx_dy, &W_H_real[Grid_fine->N], 2*Grid_fine->N, cudaMemcpyDeviceToHost);
//
//	// now compute actual palinstrophy and add everything together
//    for(int i = 0; i < Grid_fine->N; i+=1){
//		*Pal += 0.5 * (Grid_fine->h) * (Grid_fine->h) * (Host_W_coarse_real_dx_dy[i] * Host_W_coarse_real_dx_dy[i] + Host_W_coarse_real_dx_dy[i + Grid_fine->N] * Host_W_coarse_real_dx_dy[i + Grid_fine->N]);
//	}
//}



/*******************************************************************
*							 Fourier							   *
*******************************************************************/



// Non-uniform discrete Fourier transform in 1D
void NDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N){
	
	// X_{k} = \sum_{n=0}^{N-1} x_{n} e^{-2\pi i p_n f_k} 
	// X_k is a complex in Fourier space 	; 	x_n are the values of the function in real space 	; 	p_n \in [0,1] are the sample points ; f_k \in [0,N] are frequencies
	
	for(int k = 0; k < N; k+=1){
		X_k[k].x = 0;
		X_k[k].y = 0;
		for(int n = 0; n < N; n+=1){
			// X_k[k] += x_n[n]*(exp(-2*i*PI*p_n[n]*f_k[k]))/N
			X_k[k].x +=  x_n[n]*cos(twoPI*p_n[n]*f_k[k])/N;
			X_k[k].y += -x_n[n]*sin(twoPI*p_n[n]*f_k[k])/N;
		}
	}
}


// Non-uniform inverse discrete Fourier transform in 1D
void iNDFT_1D(cufftDoubleComplex *X_k, double *x_n, double *p_n, double *f_k, int N){
	
	// X_k is a complex in Fourier space 	; 	x_n are the values of the function in real space 	; 	p_n \in [0,1] are the sample points ; f_k \in [0,N] are frequencies
	
	for(int n = 0; n < N; n+=1){
		x_n[n] = 0;
		for(int k = 0; k < N; k+=1){
			// x_n[n] += X_k[k]*(exp(2*i*PI*p_n[n]*f_k[k]))
			x_n[n] +=  X_k[k].x*cos(twoPI*p_n[n]*f_k[k]);
			x_n[n] += -X_k[k].y*sin(twoPI*p_n[n]*f_k[k]);
			/*
			x_n[n].x +=  X_k[k].x*cos(twoPI*p_n[n]*f_k[k]);
			x_n[n].x += -X_k[k].y*sin(twoPI*p_n[n]*f_k[k]);
			x_n[n].y +=  X_k[k].x*sin(twoPI*p_n[n]*f_k[k]);
			x_n[n].y +=  X_k[k].y*cos(twoPI*p_n[n]*f_k[k]);
			*/
		}
	}
}


// Non-uniform discrete Fourier transform in 2D
__global__ void NDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int NX, int Np){
	
	// X_{k} = \sum_{n=0}^{N-1} x_{n} e^{-2\pi i p_n . f_k} 
	// X_k is a complex in Fourier space 	; 	x_n are the values of the function in real space 	; 	p_n \in [0,1] are the sample points 	; 	f_k \in [0,N-1] are frequencies
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NX)
		return;
	
	int In = iY*NX + iX;
	
	X_k[In].x = 0;
	X_k[In].y = 0;
	
	for(int n = 0; n < Np; n+=1){
		// X_k[k] += x_n[n]*(exp(-2*i*PI*p_n[n]*f_k[k]))/(N*N)
		X_k[In].x +=  x_n[n] * cos(twoPI * ( p_n[2*n]*f_k[iX] + p_n[2*n+1]*f_k[iY] ));
		X_k[In].y += -x_n[n] * sin(twoPI * ( p_n[2*n]*f_k[iX] + p_n[2*n+1]*f_k[iY] ));
	}
	X_k[In].x = X_k[In].x/(NX*NX);
	X_k[In].y = X_k[In].y/(NX*NX);
	
}


// Non-uniform inverse discrete Fourier transform in 2D
__global__ void iNDFT_2D(cufftDoubleComplex *X_k, double *x_n, double *p_n, int *f_k, int N_grid){
	
	// X_k is a complex in Fourier space 	; 	x_n are the values of the function in real space 	; 	p_n \in [0,1] are the sample points 	; 	f_k \in [0,N-1] are frequencies
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	
	x_n[iX] = 0;
	
	for(int kx = 0; kx < N_grid; kx+=1){
		for(int ky = 0; ky < N_grid; ky+=1){
			x_n[iX] +=  X_k[ky*N_grid + kx].x * cos(twoPI * ( p_n[2*iX]*f_k[kx] + p_n[2*iX+1]*f_k[ky] ));
			x_n[iX] += -X_k[ky*N_grid + kx].y * sin(twoPI * ( p_n[2*iX]*f_k[kx] + p_n[2*iX+1]*f_k[ky] ));
			/*
			x_n[iX].x  =  X_k[ky*NX + kx].x * cos(twoPI * ( p_n[iX]*f_k[kx] + p_n[iX]*f_k[ky] ));
			x_n[iX].x += -X_k[ky*NX + kx].y * sin(twoPI * ( p_n[iX]*f_k[kx] + p_n[iX]*f_k[ky] ));
			x_n[iX].y  =  X_k[ky*NX + kx].x * sin(twoPI * ( p_n[iX]*f_k[kx] + p_n[iX]*f_k[ky] ));
			x_n[iX].y +=  X_k[ky*NX + kx].y * cos(twoPI * ( p_n[iX]*f_k[kx] + p_n[iX]*f_k[ky] ));
			*/
		}
	}
}


void Laplacian_vort(TCudaGrid2D *Grid_fine, double *Dev_W_fine, cufftDoubleComplex *Dev_Complex_fine, cufftDoubleComplex *Dev_Hat_fine, double *Dev_lap_fine_real, cufftDoubleComplex *Dev_lap_fine_complex, cufftDoubleComplex *Dev_lap_fine_hat, cufftHandle cufftPlan_fine){

    k_real_to_comp<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_W_fine, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
    cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, CUFFT_FORWARD);
    k_normalize<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);

    k_fft_lap<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Dev_lap_fine_hat, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
    cufftExecZ2Z(cufftPlan_fine, Dev_lap_fine_hat, Dev_lap_fine_complex, CUFFT_INVERSE);
    k_comp_to_real<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_lap_fine_complex, Dev_lap_fine_real, Grid_fine->NX, Grid_fine->NY);

}

__host__ __device__ double L1(double t, double tp, double tm, double tmm){
    return ((t-tm)*(t-tmm)/((tp-tm)*(tp-tmm)));
}

__host__ __device__ double L2(double t, double tp, double tm, double tmm){
    return ((t-tp)*(t-tmm)/((tm-tp)*(tm-tmm)));
}

__host__ __device__ double L3(double t, double tp, double tm, double tmm){
    return ((t-tp)*(t-tm)/((tmm-tp)*(tmm-tm)));
}
