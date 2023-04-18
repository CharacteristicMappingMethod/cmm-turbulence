/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
******************************************************************************************************************************/

#ifndef __CMM_SIMULATION_HOST_H__
#define __CMM_SIMULATION_HOST_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

// this file is all about host functions



double incompressibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_map, double *ChiX, double *ChiY, double *gradChi);
double invertibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_backward, TCudaGrid2D Grid_forward,
		double *ChiX_b, double *ChiY_b, double *ChiX_f, double *ChiY_f, double *abs_invert);

// map advection
void advect_using_stream_hermite(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi,
		double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y,
		double *psi, double *t, double *dt, int loop_ctr, int direction);

// sampling from mapstack - Apply mapstacks and init from initial conditions for different variables
void apply_map_stack(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction);
void apply_map_stack_points(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction,
		double **fluid_particles_pos_in, double *fluid_particles_pos_out,
		int fluid_particles_num, int *fluid_particles_blocks, int fluid_particles_threads);

// compute hermite with derivatives in fourier space, uniform helper function fitted for all grids to utilize only input temporary variable
void fourier_hermite(TCudaGrid2D Grid, cufftDoubleComplex *Dev_In, double *Dev_Out, cufftHandle cufft_plan);

// Compute fine vorticity hermite
void translate_initial_condition_through_map_stack(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_discrete, MapStack Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufft_plan_fine_D2Z, cufftHandle cufft_plan_fine_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *W_initial, int simulation_num_c, bool initial_discrete);

// computation of psi - sampling vorticity from fine vorticity hermite as initial condition
void evaluate_stream_hermite(TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_vort,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *Psi_real,
		cufftHandle cufftPlan_coarse_Z2D, cufftHandle cufftPlan_psi, cufftHandle cufftPlan_vort,
		cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi);

// sample psi on a fixed grid with vorticity known
void psi_upsampling(TCudaGrid2D Grid, double *Dev_W,cufftDoubleComplex *Dev_Temp_C1,
		double *Dev_Psi, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);

// compute laplacian
void laplacian(TCudaGrid2D Grid, double *Dev_W, double *Dev_out, cufftDoubleComplex *Dev_Temp_C1,
		cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);
// compute x-gradient
void grad_x(TCudaGrid2D Grid, double *Dev_W, double *Dev_out, cufftDoubleComplex *Dev_Temp_C1,
		cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);
// compute y-gradient
void grad_y(TCudaGrid2D Grid, double *Dev_W, double *Dev_out, cufftDoubleComplex *Dev_Temp_C1,
		cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);

// Computation of Global conservation values
std::string compute_conservation_targets(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufft_plan_coarse_D2Z, cufftHandle cufft_plan_coarse_Z2D, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D,
		cufftDoubleComplex *Dev_Temp_C1);

// Sample on a specific grid and save everything
std::string sample_compute_and_write(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		MapStack Map_Stack, MapStack Map_Stack_f, TCudaGrid2D* Grid_sample, TCudaGrid2D Grid_discrete, double *Dev_sample,
		cufftHandle* cufft_plan_sample_D2Z, cufftHandle* cufft_plan_sample_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double **Host_forward_particles_pos, double **Dev_forward_particles_pos, int* forward_particles_block, int forward_particles_thread,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f,
		double *bounds, double *W_initial_discrete);

// sample vorticity with mapstack at arbitrary frame
std::string Zoom(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		MapStack Map_Stack, MapStack Map_Stack_f, TCudaGrid2D* Grid_zoom, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_discrete,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f,
		double *Dev_Temp, double *W_initial_discrete, double *psi,
		double **Host_particles_pos, double **Dev_particles_pos,
		double **Host_forward_particles_pos, double **Dev_forward_particles_pos, int* forward_particles_block, int forward_particles_thread);

// check all targets to compute the next intermediate step
double compute_next_timestep(SettingsCMM SettingsMain, double t_now, double dt_now);

//void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);

#endif
