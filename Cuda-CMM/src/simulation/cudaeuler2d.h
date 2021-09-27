#ifndef __CUDA_EULER_2D_H__
#define __CUDA_EULER_2D_H__

#include "../simulation/settings.h"
#include "../simulation/cudasimulation2d.h"
#include "../grid/cudagrid2d.h"
#include "../particle/cudaparticle2d.h"
#include "../simulation/cudamesure2d.h"

#ifdef __CUDACC__

//calculate a new inital condition for hermite	by applying full map stack
void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *Dev_ChiX_stack, double *Dev_ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *Dev_ChiX, double *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, double *W_real, double *W_H_real, cufftHandle cufftPlan_fine, double *W_initial, int simulation_num, cufftDoubleComplex *Dev_Complex_fine, cufftDoubleComplex *Dev_Hat_fine, cufftDoubleComplex *Dev_Hat_fine_bis);
void evaluate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_set, double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_set, cufftDoubleComplex *Dev_Complex_coarse, cufftDoubleComplex *Dev_Hat_coarse, cufftDoubleComplex *Dev_Hat_coarse_bis, int molly_stencil);
void compute_conservation_targets(TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi, double *Dev_Psi, double *Host_Psi, double *Dev_W_coarse, double *Host_W_coarse, double *Dev_W_fine, double *Host_W_fine, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_fine, cufftDoubleComplex *Dev_Complex, cufftDoubleComplex *Dev_Hat, cufftDoubleComplex *Dev_Hat_bis, double *Mesure, double *Mesure_fine, int count_mesure);
//cudaStream_t streams

#endif

void cuda_euler_2d(SettingsCMM SettingsMain, double final_time_override = -1, double time_step_factor = 1);

string create_directory_structure(string simulationName, int NXc, int NXf, double dt, double T, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length, double inCompThreshold, string time_integration);

void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);

#endif
