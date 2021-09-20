#ifndef __CUDA_EULER_2D_H__
#define __CUDA_EULER_2D_H__

#include "../simulation/cudasimulation2d.h"
#include "../grid/cudagrid2d.h"
#include "../particle/cudaparticle2d.h"
#include "../simulation/cudamesure2d.h"

#ifdef __CUDACC__

//calculate a new inital condition for hermite	by applying full map stack
void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX_stack, ptype *Dev_ChiY_stack, ptype *Host_ChiX_stack_RAM_0, ptype *Host_ChiY_stack_RAM_0, ptype *Host_ChiX_stack_RAM_1, ptype *Host_ChiY_stack_RAM_1, ptype *Host_ChiX_stack_RAM_2, ptype *Host_ChiY_stack_RAM_2, ptype *Host_ChiX_stack_RAM_3, ptype *Host_ChiY_stack_RAM_3, ptype *Dev_ChiX, ptype *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, ptype *W_real, ptype *W_H_real, cufftHandle cufftPlan_fine, ptype *W_initial, int simulation_num, cuPtype *Dev_Complex_fine, cuPtype *Dev_Hat_fine, cuPtype *Dev_Hat_fine_bis);
void evaulate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX, ptype *Dev_ChiY, ptype *Dev_W_H_fine_real, ptype *W_real, ptype *Psi_real, cufftHandle cufftPlan_coarse, cuPtype *Dev_Complex_coarse, cuPtype *Dev_Hat_coarse, cuPtype *Dev_Hat_coarse_bis);
//cudaStream_t streams

#endif

void cuda_euler_2d(string problem_name, int grid_scale, int fine_grid_scale, string time_integration,double final_time_override = -1, double time_step_factor = 1);
string create_directory_structure(string simulationName, int NXc, int NXf, double dt, double T, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length, double inCompThreshold, string time_integration);

void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);

#endif
