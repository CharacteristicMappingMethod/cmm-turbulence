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

#ifndef __CMM_SIMULATION_HOST_H__
#define __CMM_SIMULATION_HOST_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/cmm-io.h"
#include "../ui/settings.h"

// this file is all about host functions

double incompressibility_check(TCudaGrid2D Grid_check, CmmVar2D ChiX, CmmVar2D ChiY, double *gradChi);
double invertibility_check(TCudaGrid2D Grid_check, CmmVar2D ChiX_b, CmmVar2D ChiY_b, CmmVar2D ChiX_f, CmmVar2D ChiY_f, double *abs_invert);

// map advection
void advect_using_stream_hermite(SettingsCMM SettingsMain, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Psi,
		double *Chi_new_X, double *Chi_new_Y, double *t, double *dt, int loop_ctr, int direction);

// sampling from mapstack - Apply mapstacks and init from initial conditions for different variables
void apply_map_stack(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction);
void apply_map_stack_points(SettingsCMM SettingsMain, TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY,
		double *Dev_Temp, int direction, std::map<std::string, CmmPart*> cmmPartMap, double *fluid_particles_pos_out);

// Compute fine vorticity hermite
void translate_initial_condition_through_map_stack(MapStack Map_Stack, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Vort_fine_init,
		CmmVar2D Vort_discrete_init, cufftDoubleComplex *Dev_Temp_C1, int simulation_num_c, bool initial_discrete, int var_num=0);

// computation of psi - sampling vorticity from fine vorticity hermite as initial condition
void evaluate_stream_hermite(CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Vort_fine_init, CmmVar2D Psi, CmmVar2D empty_vort,
		cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi);

void evaluate_potential_from_density_hermite(SettingsCMM SettingsMain, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Vort_fine_init, CmmVar2D Psi,
		CmmVar2D empty_vort, cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi);


// compute psi on a given Grid_psi from distribution function defined on Grid using a 1D fft
void get_psi_from_distribution_function(SettingsCMM SettingsMain, CmmVar2D Psi, CmmVar2D empty_vort, double *Dev_f_in, cufftDoubleComplex *Dev_Temp_C1);
void get_psi_hermite_from_distribution_function(SettingsCMM SettingsMain, CmmVar2D Psi, CmmVar2D empty_vort, double *Dev_f_in, cufftDoubleComplex *Dev_Temp_C1);



// all save functions under one roof
void save_functions(SettingsCMM& SettingsMain, Logger& logger, double t_now, double dt_now, double dt,
		MapStack Map_Stack, MapStack Map_Stack_f, std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap,
		cufftDoubleComplex *Dev_Temp_C1);

// Computation of Global conservation values
std::string compute_conservation_targets(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		std::map<std::string, CmmVar2D*> cmmVarMap, cufftDoubleComplex *Dev_Temp_C1);

// Sample on a specific grid and save everything
std::string sample_compute_and_write(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, MapStack Map_Stack, MapStack Map_Stack_f,
		std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap);

std::string sample_compute_and_write_vlasov(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, MapStack Map_Stack, MapStack Map_Stack_f,
		std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap);

// sample vorticity with mapstack at arbitrary frame
std::string compute_zoom(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, MapStack Map_Stack, MapStack Map_Stack_f,
		std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap);

// read map stack involves computing initial condition field so it has to be here and not in cmm-io.h
std::string readMapStack(SettingsCMM SettingsMain, MapStack& Map_Stack, std::map<std::string, CmmVar2D*> cmmVarMap,
		bool isForward, std::string data_name);

// check all targets to compute the next intermediate step
double compute_next_timestep(SettingsCMM SettingsMain, double t_now, double dt_now);

//void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);

#endif
