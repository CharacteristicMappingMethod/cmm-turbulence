#include "cudamesure2d.h"


void Compute_Energy(double *E, double *psi, TCudaGrid2D Grid){
	// parallel reduction using thrust
	thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(psi);
	*E = 0.5*Grid.h*Grid.h * thrust::transform_reduce(psi_ptr + Grid.N, psi_ptr + 3*Grid.N, thrust::square<double>(), 0.0, thrust::plus<double>());
}


void Compute_Enstrophy(double *E, double *W, TCudaGrid2D Grid){
	// parallel reduction using thrust
	thrust::device_ptr<double> W_ptr = thrust::device_pointer_cast(W);
	*E = 0.5*Grid.h*Grid.h * thrust::transform_reduce(W_ptr, W_ptr + Grid.N, thrust::square<double>(), 0.0, thrust::plus<double>());
}


// compute palinstrophy using fourier transformations - a bit expensive with two temporary arrays but ca marche
void Compute_Palinstrophy(TCudaGrid2D Grid, double *Pal, double *W_real, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, cufftHandle cufftPlan){
	// round 1: dx dervative
	k_real_to_comp<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(W_real, Dev_Temp_C1, Grid.NX, Grid.NY);
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, Grid.NX, Grid.NY);
	k_fft_dx<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid.NX, Grid.NY, Grid.h);
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);

	comp_to_real(Dev_Temp_C2, (cufftDoubleReal*)Dev_Temp_C1, Grid.N);

	// parallel reduction using thrust
	thrust::device_ptr<double> Pal_ptr = thrust::device_pointer_cast((cufftDoubleReal*)Dev_Temp_C1);
	*Pal = 0.5*Grid.h*Grid.h * thrust::transform_reduce(Pal_ptr, Pal_ptr + Grid.N, thrust::square<double>(), 0.0, thrust::plus<double>());

	// round 2: dy dervative
	k_real_to_comp<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(W_real, Dev_Temp_C1, Grid.NX, Grid.NY);
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, Grid.NX, Grid.NY);
	k_fft_dy<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid.NX, Grid.NY, Grid.h);
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);

	comp_to_real(Dev_Temp_C2, (cufftDoubleReal*)Dev_Temp_C1, Grid.N);

	// parallel reduction using thrust
	*Pal += 0.5*Grid.h*Grid.h * thrust::transform_reduce(Pal_ptr, Pal_ptr + Grid.N, thrust::square<double>(), 0.0, thrust::plus<double>());

//	printf("Pal : %f\n", *Pal);
}


//// compute palinstrophy using hermite array - cheap, but we need the vorticity hermite for that, only works for fine array, can be kernelized too actually, but not now
//void Compute_Palinstrophy_hermite(TCudaGrid2D *Grid_fine, double *Pal, double *W_H_real){
//
//	// dx and dy values are in position 2/4 and 3/4, copy to host
//	double *Host_W_coarse_real_dx_dy = new double[2*Grid_fine.N];
//	cudaMemcpy(Host_W_coarse_real_dx_dy, &W_H_real[Grid_fine.N], 2*Grid_fine.N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(Host_W_coarse_real_dx_dy, &W_H_real[Grid_fine.N], 2*Grid_fine.N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(Host_W_coarse_real_dx_dy, &W_H_real[Grid_fine.N], 2*Grid_fine.N, cudaMemcpyDeviceToHost);
//
//	// now compute actual palinstrophy and add everything together
//    for(int i = 0; i < Grid_fine.N; i+=1){
//		*Pal += 0.5 * (Grid_fine.h) * (Grid_fine.h) * (Host_W_coarse_real_dx_dy[i] * Host_W_coarse_real_dx_dy[i] + Host_W_coarse_real_dx_dy[i + Grid_fine.N] * Host_W_coarse_real_dx_dy[i + Grid_fine.N]);
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


void Laplacian_vort(TCudaGrid2D Grid_fine, double *Dev_W_fine, cufftDoubleComplex *Dev_Complex_fine, cufftDoubleComplex *Dev_Hat_fine, double *Dev_lap_fine_real, cufftDoubleComplex *Dev_lap_fine_complex, cufftDoubleComplex *Dev_lap_fine_hat, cufftHandle cufftPlan_fine){

    real_to_comp(Dev_W_fine, Dev_Complex_fine, Grid_fine.N);
    cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, CUFFT_FORWARD);
    k_normalize<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);

    k_fft_lap<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Hat_fine, Dev_lap_fine_hat, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);
    cufftExecZ2Z(cufftPlan_fine, Dev_lap_fine_hat, Dev_lap_fine_complex, CUFFT_INVERSE);
    comp_to_real(Dev_lap_fine_complex, Dev_lap_fine_real, Grid_fine.N);

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
