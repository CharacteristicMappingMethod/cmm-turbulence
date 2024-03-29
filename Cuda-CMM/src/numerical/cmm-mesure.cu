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

#include "../numerical/cmm-mesure.h"

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

// hashing algorithm
#include "murmur3.cu"

#include "stdio.h"


// assume Psi is in Hermite form
void Compute_Energy_H(double &E, CmmVar2D Psi, size_t offset_start){
	// parallel reduction using thrust
	thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(Psi.Dev_var+offset_start);
	E = 0.5*Psi.Grid->hx*Psi.Grid->hy * thrust::transform_reduce(psi_ptr + Psi.Grid->N, psi_ptr + 3*Psi.Grid->N, thrust::square<double>(), 0.0, thrust::plus<double>());
//	printf("Energ : %f\n", *E);
}
// similar to palinstrophy: compute by FFT here
void Compute_Energy(double &E, CmmVar2D Psi, cufftDoubleComplex *Dev_Temp_C1, size_t offset_start){
	// round 1: dx dervative
	grad_x(Psi, (cufftDoubleReal*)Dev_Temp_C1 + Psi.Grid->Nfft*2, Dev_Temp_C1, offset_start);
	// parallel reduction using thrust working on Grid.Nfft*2 due to inline fft padding
	thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast((cufftDoubleReal*)Dev_Temp_C1);
	E = 0.5*Psi.Grid->hx*Psi.Grid->hy * thrust::transform_reduce(psi_ptr + 2*Psi.Grid->Nfft, psi_ptr + 2*Psi.Grid->Nfft + Psi.Grid->N, thrust::square<double>(), 0.0, thrust::plus<double>());

	// round 2: dy dervative
	grad_y(Psi, (cufftDoubleReal*)Dev_Temp_C1 + Psi.Grid->Nfft*2, Dev_Temp_C1, offset_start);
	// parallel reduction using thrust
	E += 0.5*Psi.Grid->hx*Psi.Grid->hy * thrust::transform_reduce(psi_ptr + 2*Psi.Grid->Nfft, psi_ptr + 2*Psi.Grid->Nfft + Psi.Grid->N, thrust::square<double>(), 0.0, thrust::plus<double>());
//	printf("Energ : %f\n", *E);
}


void Compute_Enstrophy(double &E, CmmVar2D Vort, size_t offset_start){
	// parallel reduction using thrust
	thrust::device_ptr<double> W_ptr = thrust::device_pointer_cast(Vort.Dev_var+offset_start);
	E = 0.5*Vort.Grid->hx*Vort.Grid->hy * thrust::transform_reduce(W_ptr, W_ptr + Vort.Grid->N, thrust::square<double>(), 0.0, thrust::plus<double>());
//	printf("Enstr : %f\n", *E);
}

// function to get square for complex values
struct compl_square
{
    __host__ __device__
        double operator()(const cufftDoubleComplex &x) const {
            return x.x*x.x + x.y*x.y;
        }
};
void Compute_Enstrophy_fourier(double &E, cufftDoubleComplex *W, TCudaGrid2D Grid){
	// parallel reduction using thrust
	thrust::device_ptr<cufftDoubleComplex> W_ptr = thrust::device_pointer_cast(W);
	E = twoPI * twoPI * thrust::transform_reduce(W_ptr, W_ptr + Grid.Nfft, compl_square(), 0.0, thrust::plus<double>());
//	printf("Enstr : %f\n", *E);
}


// compute palinstrophy using fourier transformations
// Dev_Temp_C1 needs size 2 x Grid Vort Nfft in space
void Compute_Palinstrophy(double &Pal, CmmVar2D Vort, cufftDoubleComplex *Dev_Temp_C1, size_t offset_start){
	// round 1: dx dervative
	grad_x(Vort, (cufftDoubleReal*)Dev_Temp_C1 + Vort.Grid->Nfft*2, Dev_Temp_C1, offset_start);
	// parallel reduction using thrust working on Grid.Nfft*2 due to inline fft padding
	thrust::device_ptr<double> Pal_ptr = thrust::device_pointer_cast((cufftDoubleReal*)Dev_Temp_C1);
	Pal = 0.5*Vort.Grid->hx*Vort.Grid->hy * thrust::transform_reduce(Pal_ptr + 2*Vort.Grid->Nfft, Pal_ptr + 2*Vort.Grid->Nfft + Vort.Grid->N, thrust::square<double>(), 0.0, thrust::plus<double>());

	// round 2: dy dervative
	grad_y(Vort, (cufftDoubleReal*)Dev_Temp_C1 + Vort.Grid->Nfft*2, Dev_Temp_C1, offset_start);
	// parallel reduction using thrust
	Pal += 0.5*Vort.Grid->hx*Vort.Grid->hy * thrust::transform_reduce(Pal_ptr + 2*Vort.Grid->Nfft, Pal_ptr + 2*Vort.Grid->Nfft + Vort.Grid->N, thrust::square<double>(), 0.0, thrust::plus<double>());
//	printf("Pal : %f\n", *Pal);
}

/*
##############################################################################
Vlasov Poisson 
##############################################################################
*/ 

void Compute_Mass(double &mass, CmmVar2D VarIn, size_t offset_start) {
	// #############################################################################################
	//   this function computes the mass of the distribution function
	//		 mass(t) =\frac{1}{2}\int_{\Omega_v} \int_{\Omega_x} f(x,v,t) \d x\, \d v\,.
	// #############################################################################################
	
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(VarIn.Dev_var+offset_start);
	mass = VarIn.Grid->hy * VarIn.Grid->hx * thrust::reduce( ptr, ptr + VarIn.Grid->N, 0.0, thrust::plus<double>());
}

__global__ void generate_gridvalues_f_times_v(cufftDoubleReal *v2,cufftDoubleReal *f, TCudaGrid2D Grid) {
	// #############################################################################################
	// This function creates a NX time NV grid of values f(ix,iv) *v[iv]
	// #############################################################################################
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;

	int In;
	In = iY*Grid.NX + iX;
	double v_temp = Grid.bounds[2] + iY*Grid.hy;
	v2[In] = v_temp*f[In];
}


__global__ void generate_Esincos(cufftDoubleReal *Esincos, cufftDoubleReal *Efield, double k, TCudaGrid2D Grid) {
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	if(iX >= Grid.NX) return;

	double x = Grid.bounds[0] + iX*Grid.hx;
	Esincos[iX] = Efield[iX]*sin(twoPI * k * x / Grid.bounds[1]);
    Esincos[iX+Grid.NX] = Efield[iX]*cos(twoPI * k * x / Grid.bounds[1]);
}



__global__ void generate_gridvalues_f_times_v_square(cufftDoubleReal *v2,cufftDoubleReal *f, TCudaGrid2D Grid) {
	// #############################################################################################
	// This function creates a NX time NV grid of values f(ix,iv) *v[iv]^2
	// #############################################################################################
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;

	int In;
	In = iY*Grid.NX + iX;
	double v_temp = Grid.bounds[2] + iY*Grid.hy;
	v2[In] = v_temp*v_temp*f[In];
}

void Compute_Momentum(double &P_out, CmmVar2D VarIn, cufftDoubleReal *Dev_Temp_C1, size_t offset_start){
	// #############################################################################################
	//   this function computes the electrical energy of the system defined by the potential psi
	//		 P(t) = \int_{\Omega_v} \int_{\Omega_x} f(x,v,t) v \d x\, \d v\,.
	// #############################################################################################
	
	generate_gridvalues_f_times_v<<<VarIn.Grid->blocksPerGrid, VarIn.Grid->threadsPerBlock>>>(Dev_Temp_C1, VarIn.Dev_var+offset_start, *VarIn.Grid);
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Dev_Temp_C1);
	P_out = VarIn.Grid->hy * VarIn.Grid->hx * thrust::reduce( ptr, ptr + VarIn.Grid->N, 0.0, thrust::plus<double>());
}


void Compute_Kinetic_Energy(double &E_out, CmmVar2D VarIn, cufftDoubleReal *Dev_Temp_C1, size_t offset_start){
	// #############################################################################################
	//   this function computes the electrical energy of the system defined by the potential psi
	//		 \Ekin(t) =\frac{1}{2}\int_{\Omega_v} \int_{\Omega_x} f(x,v,t) \abs{v}^2 \d x\, \d v\,.
	// #############################################################################################
	
	generate_gridvalues_f_times_v_square<<<VarIn.Grid->blocksPerGrid, VarIn.Grid->threadsPerBlock>>>(Dev_Temp_C1, VarIn.Dev_var+offset_start, *VarIn.Grid);
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Dev_Temp_C1);
	E_out = 0.5 * VarIn.Grid->hy * VarIn.Grid->hx * thrust::reduce( ptr, ptr + VarIn.Grid->N, 0.0, thrust::plus<double>());
}

void Compute_Potential_Energy(double &E_out, CmmVar2D VarIn, size_t offset_start){
	// #############################################################################################
	// this function computes the electrical energy of the system defined by the potential psi
	// 		 \Epot(t) =\frac{1}{2}\int_{\Omega_x} \abs{E(x,t)}^2 \d x \,.
	// 	 where E(x,t) = -\frac{\partial \psi(x,t)}{\partial x}
	// #############################################################################################

	// parallel reduction using thrust
	thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(VarIn.Dev_var+offset_start);
	E_out = 0.5 * VarIn.Grid->hx * thrust::transform_reduce(psi_ptr + 1*VarIn.Grid->N, psi_ptr + VarIn.Grid->N + VarIn.Grid->NX, thrust::square<double>(), 0.0, thrust::plus<double>());
}

void Compute_kth_Mode(double *Emodes, CmmVar2D VarIn, cufftDoubleComplex *Dev_Temp_C1, int Nf){
	// #############################################################################################
	// this function computes the absolut value of the first Nf fourier modes of the electric field
	// #############################################################################################
   
    cufftDoubleComplex Temp_Host[Nf+2];

	cufftExecD2Z (VarIn.plan_1D_D2Z, VarIn.Dev_var + 1*VarIn.Grid->N, (cufftDoubleComplex*) (Dev_Temp_C1));
	// devide by NX to normalize FFT
	k_normalize_1D_hx<<<VarIn.Grid->fft_blocks.x, VarIn.Grid->threadsPerBlock.x>>>((cufftDoubleComplex*) (Dev_Temp_C1),  *VarIn.Grid);  // this is a normalization factor of FFT? if yes we dont need to do it everytime!!!
    cudaMemcpy(Temp_Host, Dev_Temp_C1, (Nf+1) * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < Nf; i++) {
        Emodes[i] = sqrt(Temp_Host[i+1].x*Temp_Host[i+1].x + Temp_Host[i+1].y*Temp_Host[i+1].y);
        Emodes[i] /= 1;//(double)(VarIn.Grid->NX);
    }
     //error("gabuuuu", 124);
}


void Compute_Total_Energy(double &E_tot, double &E_kin, double &E_pot, CmmVar2D VarKin, CmmVar2D VarPot, cufftDoubleReal *Dev_Temp_C1, size_t offset_start_K, size_t offset_start_P){
	// #############################################################################################
	//  this function computes the total energy of the vlasov poisson system defined by the potential psi and the distribution function f
	// #############################################################################################
	Compute_Kinetic_Energy(E_kin, VarKin, Dev_Temp_C1, offset_start_K);
	Compute_Potential_Energy(E_pot, VarPot, offset_start_P);
	E_tot = E_kin + E_pot;
}


// function to compute several global quantities for comparison purposes
// currently: min, max, area weighted avg, area weighted L2 norm
void Compute_min(double &mesure, CmmVar2D Var, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Var.Dev_var+offset_start);
	mesure = *thrust::min_element(ptr, ptr + Var.Grid->N);
}
void Compute_max(double &mesure, CmmVar2D Var, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Var.Dev_var+offset_start);
	mesure = *thrust::max_element(ptr, ptr + Var.Grid->N);
}
void Compute_avg(double &mesure, CmmVar2D Var, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Var.Dev_var+offset_start);
	mesure = Var.Grid->hy * Var.Grid->hx * thrust::reduce( ptr, ptr + Var.Grid->N, 0.0, thrust::maximum<double>());
}
void Compute_L2(double &mesure, CmmVar2D Var, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Var.Dev_var+offset_start);
	mesure = Var.Grid->hy * Var.Grid->hx * thrust::reduce( ptr, ptr + Var.Grid->N, 0.0, thrust::plus<double>());
}
// for normal arrays, averaged later is over length
void Compute_min(double &mesure, double* Dev_var, size_t N, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Dev_var+offset_start);
	mesure = *thrust::min_element(ptr, ptr + N);
}
void Compute_max(double &mesure, double* Dev_var, size_t N, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Dev_var+offset_start);
	mesure = *thrust::max_element(ptr, ptr + N);
}
void Compute_avg(double &mesure, double* Dev_var, size_t N, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Dev_var+offset_start);
	mesure = thrust::reduce( ptr, ptr + N, 0.0, thrust::maximum<double>()) / (double)(N);
}
void Compute_L2(double &mesure, double* Dev_var, size_t N, size_t offset_start) {
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(Dev_var+offset_start);
	mesure = thrust::reduce( ptr, ptr + N, 0.0, thrust::plus<double>()) / (double)(N);
}


// Hash an array
__global__ void k_hash_array(size_t n, int thread_num, double *in, char *out) {
    int iN = (blockIdx.x * blockDim.x + threadIdx.x);
    long long int hash_size = n/thread_num;
    if (iN * hash_size > n) return;
    if ((iN + 1) * hash_size > n) hash_size = n - iN * hash_size;  // check to match exactly the array size

    // now do the hashing
    MurmurHash3_x64_128((char*)(in + (iN * thread_num)), hash_size*8, 0, out + (iN * 16));

//	printf("Hash in kernel - ");
//	for (int i=0; i<16; i++) printf("%c", out[i]);
//	printf("\n");
}

void Hash_array(char *Hash, double *Dev_var, size_t n) {
	char *d_hash; cudaMalloc((void**)&d_hash, 16);
	k_hash_array<<<1, 1>>>(n, 1, Dev_var, d_hash);
	cudaMemcpy(Hash, d_hash, 16*sizeof(char), cudaMemcpyDeviceToHost);

//	printf("Hash in host - ");
//	for (int i=0; i<2; i++) printf("%c", Hash[i]);
//	printf("\n");
}




// void integral_v(double *distribution_function, double *density, TCudaGrid2D Grid){
// 	// parallel reduction using thrust
// 	thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(distribution_function);
// 	for (int ix = 0; ix < nx; ix++) {
// 	 	int In = iy*nx + IDX;
// 	 	v_r[IDX] += v[In];
// 	 }
// 	E = Grid.hy * thrust::transform_reduce(psi_ptr + Grid.N, psi_ptr + 3*Grid.N, thrust::square<double>(), 0.0, thrust::plus<double>());

// }


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
