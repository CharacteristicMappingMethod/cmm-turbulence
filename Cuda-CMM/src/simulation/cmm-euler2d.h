#ifndef __CUDA_EULER_2D_H__
#define __CUDA_EULER_2D_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"


// little function being used at beginning to compute norm of velocity
struct norm_fun
{
    __host__ __device__
        double operator()(const double &x1, const double &x2) const {
            return sqrt(x1*x1 + x2*x2);
        }
};

// little function to format milliseconds to readable format
string format_duration(double sec);

void fourier_hermite(TCudaGrid2D Grid, cufftDoubleComplex *Dev_In, double *Dev_Out, cufftHandle cufft_plan);

void translate_initial_condition_through_map_stack(TCudaGrid2D Grid_fine, MapStack Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufft_plan_fine_D2Z, cufftHandle cufft_plan_fine_Z2D, double *W_initial, int simulation_num_c,
		cufftDoubleComplex *Dev_Temp_C1);

void evaluate_stream_hermite(TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_vort,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real,
		cufftHandle cufftPlan_coarse_Z2D, cufftHandle cufftPlan_psi, cufftHandle cufftPlan_vort,
		cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi);

void psi_upsampling(TCudaGrid2D Grid, double *Dev_W,cufftDoubleComplex *Dev_Temp_C1,
		double *Dev_Psi, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D);

void compute_conservation_targets(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufft_plan_coarse_D2Z, cufftHandle cufft_plan_coarse_Z2D, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D,
		cufftDoubleComplex *Dev_Temp_C1, double *Mesure, double *Mesure_fine, int count_mesure);

void sample_compute_and_write(MapStack Map_Stack, TCudaGrid2D Grid_sample, double *Host_sample, double *Dev_sample,
		cufftHandle cufft_plan_sample_D2Z, cufftHandle cufft_plan_sample_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *Dev_ChiX, double*Dev_ChiY, double *bounds, double *W_initial, SettingsCMM SettingsMain, string i_num,
		double *Mesure_sample, int count_mesure);

void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb);
void Zoom(SettingsCMM SettingsMain, MapStack Map_Stack, TCudaGrid2D Grid_zoom, TCudaGrid2D Grid_psi,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_Temp, double *W_initial, double *psi,
		double *Host_particles_pos, double *Dev_particles_pos, double *Host_debug, string i_num);


// main function
void cuda_euler_2d(SettingsCMM SettingsMain);


#endif
