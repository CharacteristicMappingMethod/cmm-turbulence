#include "cudamesure2d.h"
#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"




__global__ void Compute_Energy(ptype *E, ptype *psi, int N, int NX, int NY, ptype h){


    int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NX)
		return;

	int In = iY*NX + iX;

    atomicAdd(E, 0.5*h * h * (psi[In + N] * psi[In + N] + psi[In + 2 * N] * psi[In + 2 * N]));

/*	int idx = threadIdx.x;
	int stride_x = blockDim.x;

	for(int i = idx; i < N; i+=stride_x){
		atomicAdd(E, 0.5*h * h * (psi[i + N] * psi[i + N] + psi[i + 2 * N] * psi[i + 2 * N]));
	}
    */
}


__global__ void Compute_Enstrophy(ptype *Ens, ptype *W, int N, int NX, int NY, ptype h){

    int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NX || iY >= NX)
		return;

	int In = iY*NX + iX;

    atomicAdd(Ens, 0.5 * h * h * (W[In] * W[In]));


/*	int idx = threadIdx.x;
	int stride_x = blockDim.x;

	for(int i = idx; i < N; i+=stride_x){
		atomicAdd(Ens, 0.5 * h * h * (W[i] * W[i]));
	}
    */
}


void Compute_Palinstrophy(TCudaGrid2D *Grid_coarse, ptype *Pal, ptype *W_real, cuPtype *Dev_Complex, cuPtype *Dev_Hat, cuPtype *Dev_Hat_bis, cufftHandle cufftPlan_coarse){

	ptype *Host_W_coarse_real_dx_dy = new ptype[2*Grid_coarse->N];
	
	kernel_real_to_complex<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Hat, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat, Dev_Complex, CUFFT_FORWARD);	
	kernel_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Complex, Grid_coarse->NX, Grid_coarse->NY);
	
	kernel_fft_dx<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Complex, Dev_Hat, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h); 					// x-derivative in Fourier space			    
	kernel_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Complex, Dev_Hat_bis, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h); 				// y-derivative in Fourier space
	
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat, Dev_Complex, CUFFT_INVERSE);
    cudaDeviceSynchronize();
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_bis, Dev_Hat, CUFFT_INVERSE);
	
	cudaMemcpy(Host_W_coarse_real_dx_dy, (cufftDoubleReal*)Dev_Complex, Grid_coarse->N, cudaMemcpyDeviceToHost);
	cudaMemcpy(&Host_W_coarse_real_dx_dy[Grid_coarse->N], (cufftDoubleReal*)Dev_Hat, Grid_coarse->N, cudaMemcpyDeviceToHost);


    for(int i = 0; i < Grid_coarse->N; i+=1){
		*Pal = *Pal +  (Grid_coarse->h) * (Grid_coarse->h) * (Host_W_coarse_real_dx_dy[i] * Host_W_coarse_real_dx_dy[i] + Host_W_coarse_real_dx_dy[i + Grid_coarse->N] * Host_W_coarse_real_dx_dy[i + Grid_coarse->N]);
	}

	*Pal = 0.5*(*Pal);
	
}



/*******************************************************************
*							 Fourier							   *
*******************************************************************/



// Non-uniform discrete Fourier transform in 1D
void NDFT_1D(cuPtype *X_k, ptype *x_n, ptype *p_n, ptype *f_k, int N){
	
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
void iNDFT_1D(cuPtype *X_k, ptype *x_n, ptype *p_n, ptype *f_k, int N){
	
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
__global__ void NDFT_2D(cuPtype *X_k, ptype *x_n, ptype *p_n, int *f_k, int NX, int Np){
	
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
__global__ void iNDFT_2D(cuPtype *X_k, ptype *x_n, ptype *p_n, int *f_k, int N_grid){
	
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


void Laplacian_vort(TCudaGrid2D *Grid_fine, ptype *Dev_W_fine, cuPtype *Dev_Complex_fine, cuPtype *Dev_Hat_fine, ptype *Dev_lap_fine_real, cuPtype *Dev_lap_fine_complex, cuPtype *Dev_lap_fine_hat, cufftHandle cufftPlan_fine){

    kernel_real_to_complex<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_W_fine, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
    cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, CUFFT_FORWARD);
    kernel_normalize<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);

    kernel_fft_lap<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Dev_lap_fine_hat, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
    cufftExecZ2Z(cufftPlan_fine, Dev_lap_fine_hat, Dev_lap_fine_complex, CUFFT_INVERSE);
    kernel_complex_to_real<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_lap_fine_real, Dev_lap_fine_complex, Grid_fine->NX, Grid_fine->NY);

}

__host__ __device__ ptype L1(ptype t, ptype tp, ptype tm, ptype tmm){
    return ((t-tm)*(t-tmm)/((tp-tm)*(tp-tmm)));
}

__host__ __device__ ptype L2(ptype t, ptype tp, ptype tm, ptype tmm){
    return ((t-tp)*(t-tmm)/((tm-tp)*(tm-tmm)));
}

__host__ __device__ ptype L3(ptype t, ptype tp, ptype tm, ptype tmm){
    return ((t-tp)*(t-tm)/((tmm-tp)*(tmm-tm)));
}
