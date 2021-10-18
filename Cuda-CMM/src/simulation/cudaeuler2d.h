#ifndef __CUDA_EULER_2D_H__
#define __CUDA_EULER_2D_H__

#include "../simulation/settings.h"
#include "../simulation/cmm-io.h"
#include "../simulation/cudasimulation2d.h"
#include "../grid/cudagrid2d.h"
#include "../numerical/cmm-particles.h"
#include "../simulation/cudamesure2d.h"
#include "../simulation/cmm-fft.h"

#define COPY_OLD_1
//#define COPY_OLD_2

#ifdef __CUDACC__

//calculate a new inital condition for hermite	by applying full map stack
void fourier_hermite(TCudaGrid2D *Grid, cufftDoubleComplex *Dev_Temp_C1, double *Dev_Output, cufftDoubleComplex *Dev_Temp_C2, cufftHandle cufftPlan);

void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *Dev_ChiX_stack, double *Dev_ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *Dev_ChiX, double *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, double *W_H_real, cufftHandle cufftPlan_fine, double *bounds, double *W_initial, int simulation_num, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2);

void evaluate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_set, double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_set, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, int molly_stencil, double freq_cut_psi);
void evaluate_stream_hermite_2(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_set, double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_set, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, int molly_stencil, double freq_cut_psi, double *Host_debug);
void psi_upsampling(TCudaGrid2D *Grid, double *Dev_W,cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, double *Dev_Psi, cufftHandle cufftPlan);

void compute_conservation_targets(TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi, double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_fine, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, double *Mesure, double *Mesure_fine, int count_mesure);
//cudaStream_t streams

void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);
void Zoom(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *Dev_ChiX_stack, double *Dev_ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *Dev_ChiX, double *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, double *W_real, cufftHandle cufftPlan_fine, double *W_initial, cufftDoubleComplex *Dev_Temp, string workspace, string simulationName, int simulation_num, double L);

#endif

void cuda_euler_2d(SettingsCMM SettingsMain);





template < typename Type > std::string to_str (const Type & t)
{
  std::ostringstream os;
  os << t;
  return os.str ();
}

#endif
