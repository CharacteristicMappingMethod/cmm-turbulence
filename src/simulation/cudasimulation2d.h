#ifndef __CUDA_SIMULATION_2D_H__
#define __CUDA_SIMULATION_2D_H__

#include "../hermite/cudahermite2d.h"

//fft operations testing
void test_fft_operations();

void recompute_output_files();	//read binary files storing diffeos and recompute output

#ifdef __CUDACC__


	//cpu functions
	ptype compare_map_with_identity(ptype *ChiX, ptype *ChiY, int NX, int NY, ptype h);

	//global kernel functions
	
	
	//map operations
	__global__ void kernel_init_diffeo(ptype *ChiX, ptype *ChiY, int NX, int NY, ptype h);
	__global__ void kernel_update_map_from_dual(ptype *ChiX, ptype *ChiY, ptype *X, ptype *Y, int NX, int NY, ptype ep);
	__global__ void kernel_incompressibility_check(ptype *ChiX, ptype *ChiY, ptype *gradChi, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs);	//to improve
	__global__ void kernel_compute_total_grid_Chi(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *gradChi, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs);
	__global__ void kernel_compute_enstropy_increase_rate_factors(ptype *w, ptype *phi, ptype *div1, ptype *div2, int NXc, int NYc, ptype hc, ptype ep);
	__global__ void kernel_compute_enstropy_increase_rate_factors(ptype *wHsc, ptype *ChiX, ptype *ChiY, ptype *phi, ptype *div1, ptype *div2, int NXc, int NYc, ptype hc, int NXsc, int NYsc, ptype hsc, ptype ep);
	
	//map advection 
	__global__ void kernel_advect_using_velocity_function(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep);
	__global__ void kernel_advect_using_stream_hermite(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *phi, int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep);
	__global__ void kernel_advect_using_stream_hermite2(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *phi, ptype *phi_p, int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep);
	
	
	//variable type conversion
	__global__ void kernel_real_to_complex(ptype *varR, cuPtype *varC, int NX, int NY);
	__global__ void kernel_complex_to_real(ptype *varR, cuPtype *varC, int NX, int NY);
	__global__ void kernel_real_to_complex_H(ptype *varR, cuPtype *varC, int NX, int NY);
	__global__ void kernel_complex_to_real_H(ptype *varR, cuPtype *varC, int NX, int NY);
	
	
	//map applications
	__global__ void kernel_apply_map_to_W(ptype *ChiX, ptype *ChiY, ptype *ws, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs);	//apply map on passed coarse grid to vorticiy field on sample grid
	__global__ void kernel_apply_map_stack_to_W(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ws, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs);	//apply map on passed coarse grid to vorticiy field on sample grid
	__global__ void kernel_apply_map_stack_to_W_custom(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ws, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype xl, ptype xr, ptype yl, ptype yr);	
	__global__ void kernel_apply_map_and_sample_from_hermite(ptype *ChiX, ptype *ChiY, ptype *fs, ptype *H, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, int NXh, int NYh, ptype hh);	//apply map and sample function from passed hermite
	__global__ void kernel_compare_map_stack_with_identity(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *error, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs);	
	__global__ void kernel_compare_vorticity_with_initial(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *error, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs);	
	
	//fft operations
	__global__ void kernel_sample_on_coarse_grid(cuPtype *AcOut, cuPtype *AfOut, int NXc, int NYc, ptype hc, int NXf, int NYf, ptype hf);
	__global__ void kernel_normalize(cuPtype *F, int NX, int NY);
	
	
	//inital conditions & velocity functions
	__device__ ptype device_advection_velocity_X(ptype x, ptype y, ptype t);
	__device__ ptype device_advection_velocity_Y(ptype x, ptype y, ptype t);
	__device__ ptype device_initial_W(ptype x, ptype y);
	__device__ ptype initHomeoIso(ptype x, ptype y);
	
	
	void Zoom(string simulationName, ptype L, TCudaGrid2D *Gc, TCudaGrid2D *Gsf, ptype *devChiX_stack, ptype *devChiY_stack, ptype *devChiX, ptype *devChiY, ptype *devWs, int stack_length);		//Zoom
	
	
	
	
	
#endif



#endif
