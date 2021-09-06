#ifndef __CUDA_EULER_2D_H__
#define __CUDA_EULER_2D_H__

#include "../simulation/cudasimulation2d.h"
#include "../grid/cudagrid2d.h"
#include "../particle/cudaparticle2d.h"

#ifdef __CUDACC__

//calculate a new inital condition for hermite	by applying full map stack
void translate_initial_condition_through_map_stack(TCudaGrid2D *Gc, TCudaGrid2D *Gsc, ptype *devChiX_stack, ptype *devChiY_stack, ptype *devChiX, ptype *devChiY, int stack_length, ptype *devWsf, cuPtype *devWsfC, cuPtype *devWsfOut, ptype *devWHsc, cuPtype *devWHscC, cuPtype *devWHscOut, cufftHandle cufftPlan_f);
void evaulate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX, ptype *Dev_ChiY, ptype *Dev_W_H_fine_real, ptype *W_real, cuPtype *W_complex, cuPtype *W_hat, ptype *Psi_real, cuPtype *Psi_complex, cuPtype *Psi_hat, cufftHandle cufftPlan_coarse);
//cudaStream_t streams
#endif

void cuda_euler_2d(string problem_name, int grid_scale, int fine_grid_scale, double final_time_override = -1, double time_step_factor = 1);
string create_directory_structure(string simulationName, int NXc, int NXf, double dt, double T, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length, double inCompThreshold);

void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);


#endif
