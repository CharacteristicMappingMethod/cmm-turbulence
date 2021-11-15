#ifndef __CUDA_SIMULATION_2D_H__
#define __CUDA_SIMULATION_2D_H__

#include "../grid/cudagrid2d.h"
#include "../simulation/settings.h"


//cpu functions
double compare_map_with_identity(double *ChiX, double *ChiY, int NX, int NY, double h);

//global kernel functions


//map operations
__global__ void kernel_init_diffeo(double *ChiX, double *ChiY, TCudaGrid2D Grid);
__global__ void kernel_incompressibility_check(double *ChiX, double *ChiY, double *gradChi, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse);
double incompressibility_check(double *ChiX, double *ChiY, double *gradChi, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse);

__global__ void kernel_compute_total_grid_Chi(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *gradChi, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs);
__global__ void kernel_compute_enstropy_increase_rate_factors(double *w, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, double ep);
__global__ void kernel_compute_enstropy_increase_rate_factors(double *wHsc, double *ChiX, double *ChiY, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, int NXsc, int NYsc, double hsc, double ep);

//map advection
void advect_using_stream_hermite(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, double *t, double *dt, int loop_ctr);

void advect_using_stream_hermite_grid(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, double *t, double *dt, int loop_ctr);
__global__ void k_compute_footpoints(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, int time_integration_num, int l_order);
__global__ void k_map_update(double *Chi, double *Chi_new, TCudaGrid2D Grid_map, int map_ord_1, int warp_direc);

__global__ void kernel_advect_using_velocity_function(double *ChiX, double *ChiY, double *ChiDualX, double *ChiDualY, int NXc, int NYc, double hc, double t, double dt, double ep);
__global__ void kernel_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi,
		TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num, int l_order);


//map applications

__global__ void kernel_apply_map_and_sample_from_hermite(double *ChiX, double *ChiY, double *fs, double *H, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_vort, TCudaGrid2D Grid_fine, int molly_stencil, bool padd_inline_fft);
__global__ void kernel_compare_map_stack_with_identity(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs);
__global__ void kernel_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, TCudaGrid2D Grid_map, TCudaGrid2D Grid_fine, int simulation_num);

void apply_map_stack_to_W_part_All(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY,
		double *W_real, double *Dev_Temp, double *W_initial, int simulation_num);

__global__ void kernel_apply_map_stack_to_W_custom_part_1(double *ChiX, double *ChiY, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void kernel_apply_map_stack_to_W_part_2(double *ChiX_stack, double *ChiY_stack, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void kernel_apply_map_stack_to_W_part_3(double *ws, double *x_y, TCudaGrid2D Grid, double *W_initial, int simulation_num);



//inital conditions & velocity functions
__device__ double device_advection_velocity_X(double x, double y, double t);
__device__ double device_advection_velocity_Y(double x, double y, double t);
__device__ double device_initial_W(double x, double y, int simulation_num);
__device__ double device_initial_W_discret(double x, double y, double *W_initial, int NX, int NY);
__device__ double initHomeoIso(double x, double y);

__global__ void k_h_sample_map(double *ChiX, double *ChiY, double *ChiX_s, double *ChiY_s, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_h_sample(double *val, double *val_s, TCudaGrid2D Grid, TCudaGrid2D Grid_s);


#endif
