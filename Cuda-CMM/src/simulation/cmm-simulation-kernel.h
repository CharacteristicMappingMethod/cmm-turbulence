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

#ifndef __CMM_SIMULATION_KERNEL_H__
#define __CMM_SIMULATION_KERNEL_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

// this file is all about global and device functions
extern __constant__ double d_L1[4], d_L12[4], d_c1[12], d_cx[12], d_cy[12], d_cxy[12], d_bounds[4];

// init map kernel
__global__ void k_init_diffeo(double *ChiX, double *ChiY, TCudaGrid2D Grid);
__global__ void k_init_diffeo(double2 *Chi, TCudaGrid2D Grid);

__global__ void k_swap_h(double *Val1, double *Val2, TCudaGrid2D Grid);


// sampling kernels
__global__ void k_h_sample_map(double *ChiX, double *ChiY, double *ChiX_s, double *ChiY_s, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_h_sample_points_map(TCudaGrid2D Grid_map, TCudaGrid2D Grid, double *ChiX, double *ChiY, double* Dev_particles_pos_in, double* Dev_particles_pos_out, int particles_num);
__global__ void k_h_sample(double *val, double *val_s, TCudaGrid2D Grid, TCudaGrid2D Grid_s);
__global__ void k_h_sample_points_dxdy(TCudaGrid2D Grid_map, TCudaGrid2D Grid, double *val_h, double* Dev_particles_pos_in, double* Dev_particles_pos_out, int particle_num);

// checking kernels
__global__ void k_incompressibility_check(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_map,
		double *ChiX, double *ChiY, double *grad_Chi);
__global__ void k_invertibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_backward, TCudaGrid2D Grid_forward,
		double *ChiX_b, double *ChiY_b, double *ChiX_f, double *ChiY_f, double *abs_invert);

// map update kernels
__global__ void k_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi,
		TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num, int l_order, int direction);

// apply map stack kernels to sample from initial condition
__global__ void k_h_sample_map_compact(double *ChiX, double *ChiY, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_apply_map_compact(double *ChiX_stack, double *ChiY_stack, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid);
__global__ void k_h_sample_from_init(double *ws, double *x_y, TCudaGrid2D Grid, TCudaGrid2D Grid_discrete, int init_var_num, int init_num, double *W_initial_discrete, bool initial_discrete);
__global__ void k_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, TCudaGrid2D Grid_map, TCudaGrid2D Grid_fine, int simulation_num);
__global__ void k_apply_map_and_sample_from_hermite(double *ChiX, double *ChiY, double *fs, double *H, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_vort, TCudaGrid2D Grid_fine, int molly_stencil, bool padd_inline_fft);

// vlasov integral
__global__ void integral_v(double *v_in, double *v_out, int nx, int ny, double hy);
__global__ void k_assemble_psi(double *phi_1D, double *psi_out, double *velocity, TCudaGrid2D Grid);
__global__ void generate_gridvalues_v(cufftDoubleReal *v, double prefactor, TCudaGrid2D Grid);
__global__ void copy_first_row_to_all_rows_in_grid(cufftDoubleReal *val_in, cufftDoubleReal *val_out, TCudaGrid2D Grid);
// use cudaMemset(array, value, Size) for this
//__global__ void set_value(cufftDoubleReal *array_in, double value, TCudaGrid2D Grid);

#endif
