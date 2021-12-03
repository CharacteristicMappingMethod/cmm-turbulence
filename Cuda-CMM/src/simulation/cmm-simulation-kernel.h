#ifndef __CMM_SIMULATION_KERNEL_H__
#define __CMM_SIMULATION_KERNEL_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

// this file is all about global and device functions
extern __constant__ double d_L1[4], d_L12[4], d_c1[12], d_cx[12], d_cy[12], d_cxy[12], d_bounds[4];

// init map kernel
__global__ void k_init_diffeo(double *ChiX, double *ChiY, TCudaGrid2D Grid);

__global__ void k_swap_h(double *Val1, double *Val2, TCudaGrid2D Grid);


// sampling kernels
__global__ void k_h_sample_map(double *ChiX, double *ChiY, double *ChiX_s, double *ChiY_s, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_h_sample(double *val, double *val_s, TCudaGrid2D Grid, TCudaGrid2D Grid_s);

__global__ void k_incompressibility_check(double *ChiX, double *ChiY, double *gradChi, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse);

// map update kernels
__global__ void k_compute_footpoints(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, int time_integration_num, int l_order);
__global__ void k_map_update(double *Chi, double *Chi_new, TCudaGrid2D Grid_map, int map_ord_1, int warp_direc);

__global__ void k_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi,
		TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num, int l_order);

// apply map stack kernels to sample from initial condition
__global__ void k_h_sample_map_compact(double *ChiX, double *ChiY, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_apply_map_compact(double *ChiX_stack, double *ChiY_stack, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_h_sample_from_init(double *ws, double *x_y, TCudaGrid2D Grid, TCudaGrid2D Grid_discrete, double *W_initial_discrete, int simulation_num, bool initial_discrete);

__global__ void k_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, TCudaGrid2D Grid_map, TCudaGrid2D Grid_fine, int simulation_num);

__global__ void k_apply_map_and_sample_from_hermite(double *ChiX, double *ChiY, double *fs, double *H, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_vort, TCudaGrid2D Grid_fine, int molly_stencil, bool padd_inline_fft);

// device initial functions
__device__ double d_initial_W(double x, double y, int simulation_num);
__device__ double d_initial_W_discret(double x, double y, double *W_initial, int NX, int NY);

#endif
