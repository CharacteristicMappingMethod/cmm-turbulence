#ifndef __CUDA_SIMULATION_2D_H__
#define __CUDA_SIMULATION_2D_H__

#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"

//fft operations testing
void test_fft_operations();

void recompute_output_files();	//read binary files storing diffeos and recompute output

#ifdef __CUDACC__


	//cpu functions
	double compare_map_with_identity(double *ChiX, double *ChiY, int NX, int NY, double h);

	//global kernel functions
	
	
	//map operations
	__global__ void kernel_init_diffeo(double *ChiX, double *ChiY, int NX, int NY, double h);
	__global__ void kernel_incompressibility_check(double *ChiX, double *ChiY, double *gradChi, int NXc, int NYc, double hc, int NXs, int NYs, double hs);	//to improve
	__global__ void kernel_compute_total_grid_Chi(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *gradChi, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs);
	__global__ void kernel_compute_enstropy_increase_rate_factors(double *w, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, double ep);
	__global__ void kernel_compute_enstropy_increase_rate_factors(double *wHsc, double *ChiX, double *ChiY, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, int NXsc, int NYsc, double hsc, double ep);
	
	//map advection 
	__global__ void kernel_advect_using_velocity_function(double *ChiX, double *ChiY, double *ChiDualX, double *ChiDualY, int NXc, int NYc, double hc, double t, double dt, double ep);
	__global__ void kernel_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *phi, double *phi_p, double *phi_p_p, int NXc, int NYc, double hc, int NX_psi, int NY_psi, double h_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num);
	
	
	//map applications
	void apply_map_stack_to_W_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY,
			double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1,
			double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3,
			double *W_real, cufftDoubleComplex *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM,
			int mem_RAM, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double *bounds, double *W_initial, int simulation_num);
	void copy_stack_to_device(int K_RAM, long int index, long int length_bytes, double *ChiX_stack, double *ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3);

	__global__ void kernel_apply_map_to_W(double *ChiX, double *ChiY, double *ws, int NXc, int NYc, double hc, int NXs, int NYs, double hs);	//apply map on passed coarse grid to vorticiy field on sample grid
	__global__ void kernel_apply_map_stack_to_W(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ws, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double *W_initial, int simulation_num);	//apply map on passed coarse grid to vorticiy field on sample grid
	__global__ void kernel_apply_map_stack_to_W_custom(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ws, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double xl, double xr, double yl, double yr, double *W_initial, int simulation_num);
	__global__ void kernel_apply_map_and_sample_from_hermite(double *ChiX, double *ChiY, double *fs, double *H, int NXc, int NYc, double hc, int NXs, int NYs, double hs, int NXh, int NYh, double hh, int molly_stencil);	//apply map and sample function from passed hermite
	__global__ void kernel_compare_map_stack_with_identity(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs);
	__global__ void kernel_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs, int simulation_num);
	
	__global__ void kernel_apply_map_stack_to_W_custom_part_1(double *ChiX, double *ChiY, cufftDoubleComplex *x_y, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double xl, double xr, double yl, double yr);
	__global__ void kernel_apply_map_stack_to_W_part_2(double *ChiX_stack, double *ChiY_stack, cufftDoubleComplex *x_y, int NXc, int NYc, double hc, int NXs, int NYs, int k);
	__global__ void kernel_apply_map_stack_to_W_part_3(double *ws, cufftDoubleComplex *x_y, int NXs, int NYs, double hs, double *W_initial, int simulation_num);
	
	
	
	//inital conditions & velocity functions
	__device__ double device_advection_velocity_X(double x, double y, double t);
	__device__ double device_advection_velocity_Y(double x, double y, double t);
	__device__ double device_initial_W(double x, double y, int simulation_num);
	__device__ double device_initial_W_discret(double x, double y, double *W_initial, int NX, int NY);
	__device__ double initHomeoIso(double x, double y);
	
	__global__ void k_upsample(double *ChiX, double *ChiY, double *ChiX_2048, double *ChiY_2048, int NXc, int NYc, double hc, int NXs, int NYs, double hs);
	
	
#endif



#endif
