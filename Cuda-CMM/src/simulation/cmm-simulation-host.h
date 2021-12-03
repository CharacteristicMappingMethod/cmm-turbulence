#ifndef __CMM_SIMULATION_HOST_H__
#define __CMM_SIMULATION_HOST_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

// this file is all about host functions



double incompressibility_check(double *ChiX, double *ChiY, double *gradChi, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse);

// map advection
void advect_using_stream_hermite_grid(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi,
		double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, double *t, double *dt, int loop_ctr);
void advect_using_stream_hermite(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi,
		double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, double *t, double *dt, int loop_ctr);

// sampling from mapstack - Apply mapstacks
void apply_map_stack_to_W(TCudaGrid2D Grid, TCudaGrid2D Grid_discrete, MapStack Map_Stack, double *ChiX, double *ChiY,
		double *W_real, double *Dev_Temp, double *W_initial_discrete, int simulation_num, bool initial_discrete);

// compute hermite with derivatives in fourier space, uniform helper function fitted for all grids to utilize only input temporary variable
void fourier_hermite(TCudaGrid2D Grid, cufftDoubleComplex *Dev_In, double *Dev_Out, cufftHandle cufft_plan);

// Compute fine vorticity hermite
void translate_initial_condition_through_map_stack(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_discrete, MapStack Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufft_plan_fine_D2Z, cufftHandle cufft_plan_fine_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *W_initial, int simulation_num_c, bool initial_discrete);

// computation of psi - sampling vorticity from fine vorticity hermite as initial condition
void evaluate_stream_hermite(TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_vort,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real,
		cufftHandle cufftPlan_coarse_Z2D, cufftHandle cufftPlan_psi, cufftHandle cufftPlan_vort,
		cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi);

// sample psi on a fixed grid with vorticity known
void psi_upsampling(TCudaGrid2D Grid, double *Dev_W,cufftDoubleComplex *Dev_Temp_C1,
		double *Dev_Psi, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);

// compute laplacian of vorticity
void Laplacian_vort(TCudaGrid2D Grid, double *Dev_W, double *Dev_Lap, cufftDoubleComplex *Dev_Temp_C1,
		cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);

// Computation of Global conservation values
void compute_conservation_targets(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufft_plan_coarse_D2Z, cufftHandle cufft_plan_coarse_Z2D, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D,
		cufftDoubleComplex *Dev_Temp_C1, double *Mesure, double *Mesure_fine, int count_mesure);

// Sample on a specific grid and save everything
void sample_compute_and_write(MapStack Map_Stack, TCudaGrid2D Grid_sample, TCudaGrid2D Grid_discrete, double *Host_sample, double *Dev_sample,
		cufftHandle cufft_plan_sample_D2Z, cufftHandle cufft_plan_sample_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *Dev_ChiX, double*Dev_ChiY, double *bounds, double *W_initial, SettingsCMM SettingsMain, std::string i_num,
		double *Mesure_sample, int count_mesure);

// sample vorticity with mapstack at arbitrary frame
void Zoom(SettingsCMM SettingsMain, MapStack Map_Stack, TCudaGrid2D Grid_zoom, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_discrete,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_Temp, double *W_initial, double *psi,
		double *Host_particles_pos, double *Dev_particles_pos, double *Host_debug, std::string i_num);

//void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);

#endif
