#include "cmm-simulation-host.h"
#include "cmm-simulation-kernel.h"

#include "../numerical/cmm-hermite.h"
#include "../numerical/cmm-timestep.h"

#include "../numerical/cmm-fft.h"

#include "../ui/cmm-io.h"

// debugging, using printf
#include "stdio.h"
#include <math.h>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "../numerical/cmm-mesure.h"

extern __constant__ double d_L1[4], d_L12[4], d_c1[12], d_cx[12], d_cy[12], d_cxy[12], d_bounds[4];


// function to get difference to 1 for thrust parallel reduction
struct absto1
{
    __host__ __device__
        double operator()(const double &x) const {
            return fabs(1-x);
        }
};
double incompressibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_map, double *ChiX, double *ChiY, double *grad_Chi) {
	// compute determinant of gradient and save in gradchi
	k_incompressibility_check<<<Grid_check.blocksPerGrid, Grid_check.threadsPerBlock>>>(Grid_check, Grid_map, ChiX, ChiY, grad_Chi);

	// compute maximum using thrust parallel reduction
	thrust::device_ptr<double> grad_Chi_ptr = thrust::device_pointer_cast(grad_Chi);
	return thrust::transform_reduce(grad_Chi_ptr, grad_Chi_ptr + Grid_check.N, absto1(), 0.0, thrust::maximum<double>());
}

double invertibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_backward, TCudaGrid2D Grid_forward,
		double *ChiX_b, double *ChiY_b, double *ChiX_f, double *ChiY_f, double *abs_invert) {
	// compute determinant of gradient and save in gradchi
	k_invertibility_check<<<Grid_check.blocksPerGrid, Grid_check.threadsPerBlock>>>(Grid_check, Grid_backward, Grid_forward,
			ChiX_b, ChiY_b, ChiX_f, ChiY_f, abs_invert);

	// compute maximum using thrust parallel reduction
	thrust::device_ptr<double> abs_invert_ptr = thrust::device_pointer_cast(abs_invert);
	return thrust::reduce(abs_invert_ptr, abs_invert_ptr + Grid_check.N, 0.0, thrust::maximum<double>());
}


// advect stream where footpoints are just neighbouring points
void advect_using_stream_hermite_grid(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi,
		double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y,
		double *psi, double *t, double *dt, int loop_ctr, int direction) {
	// compute lagrange coefficients from dt vector for timesteps n+dt and n+dt/2, this makes them dynamic
	double h_L1[4], h_L12[4];  // constant memory for lagrange coefficient to be computed only once
	int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;  // dt and t are shifted because of initial previous steps
	for (int i_p = 0; i_p < SettingsMain.getLagrangeOrder(); ++i_p) {
		h_L1[i_p] = get_L_coefficient(t, t[loop_ctr_l+1], loop_ctr_l, i_p, SettingsMain.getLagrangeOrder());
		h_L12[i_p] = get_L_coefficient(t, t[loop_ctr_l] + dt[loop_ctr_l+1]/2.0, loop_ctr_l, i_p, SettingsMain.getLagrangeOrder());
	}

	// copy to constant memory
	cudaMemcpyToSymbol(d_L1, h_L1, sizeof(double)*4); cudaMemcpyToSymbol(d_L12, h_L12, sizeof(double)*4);

	// first of all: compute footpoints at gridpoints, here we could speedup the first sampling of u by directly using the values, as we start at exact grid point locations
	k_compute_footpoints<<<Grid_map.blocksPerGrid, Grid_map.threadsPerBlock>>>(ChiX, ChiY, Chi_new_X, Chi_new_Y, psi,
			Grid_map, Grid_psi, t[loop_ctr_l+1], dt[loop_ctr_l+1],
			SettingsMain.getTimeIntegrationNum(), SettingsMain.getLagrangeOrder(), direction);

	// update map, x and y can be done seperately
	int shared_size = (18+2*SettingsMain.getMapUpdateOrderNum())*(18+2*SettingsMain.getMapUpdateOrderNum());  // how many points do we want to load?
	k_map_update<<<Grid_map.blocksPerGrid, Grid_map.threadsPerBlock, shared_size*sizeof(double)>>>(ChiX, Chi_new_X, Grid_map, SettingsMain.getMapUpdateOrderNum()+1, 0);
	k_map_update<<<Grid_map.blocksPerGrid, Grid_map.threadsPerBlock, shared_size*sizeof(double)>>>(ChiY, Chi_new_Y, Grid_map, SettingsMain.getMapUpdateOrderNum()+1, 1);
}


// wrapper function for map advection
void advect_using_stream_hermite(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi,
		double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y,
		double *psi, double *t, double *dt, int loop_ctr, int direction) {
	// compute lagrange coefficients from dt vector for timesteps n+dt and n+dt/2, this makes them dynamic
	double h_L1[4], h_L12[4];  // constant memory for lagrange coefficient to be computed only once
	int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;  // dt and t are shifted because of initial previous steps
	for (int i_p = 0; i_p < SettingsMain.getLagrangeOrder(); ++i_p) {
		h_L1[i_p] = get_L_coefficient(t, t[loop_ctr_l+1], loop_ctr_l, i_p, SettingsMain.getLagrangeOrder());
		h_L12[i_p] = get_L_coefficient(t, t[loop_ctr_l] + dt[loop_ctr_l+1]/2.0, loop_ctr_l, i_p, SettingsMain.getLagrangeOrder());
	}

	double h_c[3];  // constant memory for map update coefficient to be computed only once
	switch (SettingsMain.getMapUpdateOrderNum()) {
		case 2: { h_c[0] = +3.0/8.0; h_c[1] = -3.0/20.0; h_c[2] = +1.0/40.0; break; }  // 6th order interpolation
		case 1: { h_c[0] = +1.0/3.0; h_c[1] = -1.0/12.0; break; }  // 4th order interpolation
		case 0: { h_c[0] = +1.0/4.0; break; }  // 2th order interpolation
	}

	double h_c1[12], h_cx[12], h_cy[12], h_cxy[12];  // compute coefficients for each direction only once
	// already compute final coefficients with appropriate sign
	for (int i_foot = 0; i_foot < 4+4*SettingsMain.getMapUpdateOrderNum(); ++i_foot) {
		h_c1 [i_foot] = h_c[i_foot/4];
		h_cx [i_foot] = h_c[i_foot/4] * (1 - 2*((i_foot/2)%2))     / SettingsMain.getMapEpsilon() / double(i_foot/4 + 1);
		h_cy [i_foot] = h_c[i_foot/4] * (1 - 2*(((i_foot+1)/2)%2)) / SettingsMain.getMapEpsilon() / double(i_foot/4 + 1);
		h_cxy[i_foot] = h_c[i_foot/4] * (1 - 2*(i_foot%2)) / SettingsMain.getMapEpsilon() / SettingsMain.getMapEpsilon() / double(i_foot/4 + 1) / double(i_foot/4 + 1);
	}

	// copy to constant memory
	cudaMemcpyToSymbol(d_L1, h_L1, sizeof(double)*4);  cudaMemcpyToSymbol(d_L12, h_L12, sizeof(double)*4);
	cudaMemcpyToSymbol(d_c1, h_c1, sizeof(double)*12); cudaMemcpyToSymbol(d_cx, h_cx, sizeof(double)*12);
	cudaMemcpyToSymbol(d_cy, h_cy, sizeof(double)*12); cudaMemcpyToSymbol(d_cxy, h_cxy, sizeof(double)*12);

//	printf("Time - %f \t dt - %f \t TimeInt - %d \t Lagrange - %d \n", t[loop_ctr_l+1], dt[loop_ctr_l+1], SettingsMain.getTimeIntegrationNum(), SettingsMain.getLagrangeOrder());

	// now launch the kernel
	k_advect_using_stream_hermite<<<Grid_map.blocksPerGrid, Grid_map.threadsPerBlock>>>(ChiX, ChiY, Chi_new_X, Chi_new_Y,
			psi, Grid_map, Grid_psi, t[loop_ctr_l+1], dt[loop_ctr_l+1],
			SettingsMain.getMapEpsilon(), SettingsMain.getTimeIntegrationNum(),
			SettingsMain.getMapUpdateOrderNum(), SettingsMain.getLagrangeOrder(), direction);
}



/*******************************************************************
*		 Apply mapstacks to get full map to initial condition	   *
*******************************************************************/
void apply_map_stack(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction)
{
	// copy bounds to constant device memory
	cudaMemcpyToSymbol(d_bounds, Grid.bounds, sizeof(double)*4);

	k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);

	// loop over all maps in map stack, where all maps are on host system
	// this could be parallelized between loading and computing?
	// direction is important to get, wether maps are applied forwards or backwards
	for (int i_map = 0; i_map < Map_Stack.map_stack_ctr; i_map++) {
		Map_Stack.copy_map_to_device((1-direction)/2*(Map_Stack.map_stack_ctr-1) + direction*i_map);
		k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
				Dev_Temp, *Map_Stack.Grid, Grid);
	}
}


/*******************************************************************
*		 Apply mapstacks to get full map to initial condition	   *
*		    and to map specific points / particles
*******************************************************************/
void apply_map_stack_points(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction,
		double *fluid_particles_pos_in, double *fluid_particles_pos_out,
		int fluid_particles_num, int fluid_particles_blocks, int fluid_particles_threads)
{
	// copy bounds to constant device memory
	cudaMemcpyToSymbol(d_bounds, Grid.bounds, sizeof(double)*4);

	// sample map
	k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);
	// sample particles / specific points
	k_h_sample_points_map<<<fluid_particles_blocks, fluid_particles_threads>>>(*Map_Stack.Grid, ChiX, ChiY,
			fluid_particles_pos_in, fluid_particles_pos_out, fluid_particles_num);

	// loop over all maps in map stack, where all maps are on host system
	// this could be parallelized between loading and computing?
	// direction is important to get, wether maps are applied forwards or backwards
	for (int i_map = 0; i_map < Map_Stack.map_stack_ctr; i_map++) {
		Map_Stack.copy_map_to_device((1-direction)/2*(Map_Stack.map_stack_ctr-1) + direction*i_map);
		// sample map
		k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
				Dev_Temp, *Map_Stack.Grid, Grid);
		// sample particles / specific points
		k_h_sample_points_map<<<fluid_particles_blocks, fluid_particles_threads>>>(*Map_Stack.Grid, Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
				fluid_particles_pos_out, fluid_particles_pos_out, fluid_particles_num);
	}
}


/*******************************************************************
*				Compute fine vorticity hermite					   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_discrete, MapStack Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *W_initial_discrete, int simulation_num_c, bool initial_discrete)
{

	// Sample vorticity on fine grid
	// W_H_real is used as temporary variable and output
	apply_map_stack(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY, W_H_real+Grid_fine.N, -1);
	// initial condition - 0 to switch for vorticity
	k_h_sample_from_init<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(W_H_real, W_H_real+Grid_fine.N, Grid_fine, Grid_discrete,
			0, simulation_num_c, W_initial_discrete, initial_discrete);

	// go to comlex space
	cufftExecD2Z(cufftPlan_fine_D2Z, W_H_real, Dev_Temp_C1);
	k_normalize_h<<<Grid_fine.fft_blocks, Grid_fine.threadsPerBlock>>>(Dev_Temp_C1, Grid_fine);

	// cut_off frequencies at N_fine/3 for turbulence (effectively 2/3)
//	k_fft_cut_off_scale<<<Grid_fineblocksPerGrid, Grid_finethreadsPerBlock>>>(Dev_Temp_C1, Grid_fineNX, (double)(Grid_fineNX)/3.0);

	// form hermite formulation
	fourier_hermite(Grid_fine, Dev_Temp_C1, W_H_real, cufftPlan_fine_Z2D);
}



/*******************************************************************
*						 Computation of Psi						   *
*******************************************************************/
void evaluate_stream_hermite(TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_vort,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *Psi_real,
		cufftHandle cufft_plan_coarse_Z2D, cufftHandle cufft_plan_psi, cufftHandle cufft_plan_vort,
		cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi)
{

	// apply map to w and sample using mollifier, do it on a special grid for vorticity and apply mollification if wanted
	k_apply_map_and_sample_from_hermite<<<Grid_vort.blocksPerGrid, Grid_vort.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY,
			(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_fine_real, Grid_coarse, Grid_vort, Grid_fine, molly_stencil, true);

	// forward fft
	cufftExecD2Z(cufft_plan_vort, (cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C1);
	k_normalize_h<<<Grid_vort.fft_blocks, Grid_vort.threadsPerBlock>>>(Dev_Temp_C1, Grid_vort);

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3) and compute smooth W
	// use Psi grid here for intermediate storage
//	k_fft_cut_off_scale<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_Temp_C1, Grid_coarse.NX, (double)(Grid_psi.NX)/3.0);

	// transition to stream function grid with three cases : grid_vort < grid_psi, grid_vort > grid_psi (a bit dumb) and grid_vort == grid_psi
	// grid change because inline data movement is nasty, we can use psi_real as buffer anyways
	if (Grid_vort.NX != Grid_psi.NX || Grid_vort.NY != Grid_psi.NY) {
		k_fft_grid_move<<<Grid_psi.fft_blocks, Grid_psi.threadsPerBlock>>>(Dev_Temp_C1, (cufftDoubleComplex*) Psi_real, Grid_psi, Grid_vort);
	}
	// no movement needed, just copy data over
	else {
		cudaMemcpy(Psi_real, Dev_Temp_C1, Grid_vort.sizeNfft, cudaMemcpyDeviceToDevice);
	}

	// cut high frequencies in fourier space, however not that much happens after zero move add from coarse grid
	k_fft_cut_off_scale_h<<<Grid_psi.fft_blocks, Grid_psi.threadsPerBlock>>>((cufftDoubleComplex*) Psi_real, Grid_psi, freq_cut_psi);

	// Forming Psi hermite now on psi grid
	k_fft_iLap_h<<<Grid_psi.fft_blocks, Grid_psi.threadsPerBlock>>>((cufftDoubleComplex*) Psi_real, Dev_Temp_C1, Grid_psi);

	// Inverse laplacian in Fourier space
	fourier_hermite(Grid_psi, Dev_Temp_C1, Psi_real, cufft_plan_psi);
}
// debugging lines, could be needed here to check psi
//	cudaMemcpy(Host_Debug, Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(4*Grid_psi.N, Host_Debug, "psi_debug_4_nodes_C512_F2048_t64_T1", "Debug_2");


// sample psi on a fixed grid with vorticity known - assumes periodicity is preserved (no zoom!)
void psi_upsampling(TCudaGrid2D Grid, double *Dev_W, cufftDoubleComplex *Dev_Temp_C1, double *Dev_Psi,
		cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D){
	cufftExecD2Z(cufft_plan_D2Z, Dev_W, Dev_Temp_C1);
	k_normalize_h<<<Grid.fft_blocks, Grid.threadsPerBlock>>>(Dev_Temp_C1, Grid);

	// Forming Psi hermite
	k_fft_iLap_h<<<Grid.fft_blocks, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C1, Grid);
	fourier_hermite(Grid, Dev_Temp_C1, Dev_Psi, cufft_plan_Z2D);
}



// compute laplacian on variable grid, needs Grid.sizeNfft + Grid.sizeN memory
void laplacian(TCudaGrid2D Grid, double *Dev_W, double *Dev_out, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D){
    cufftExecD2Z(cufft_plan_D2Z, Dev_W, Dev_Temp_C1);
    k_normalize_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Grid);

    k_fft_lap_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C1, Grid);
    cufftExecZ2D(cufft_plan_Z2D, Dev_Temp_C1, Dev_out);
}

// compute x-gradient on variable grid, needs Grid.sizeNfft + Grid.sizeN memory
void grad_x(TCudaGrid2D Grid, double *Dev_W, double *Dev_out, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D){
    cufftExecD2Z(cufft_plan_D2Z, Dev_W, Dev_Temp_C1);
    k_normalize_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Grid);

    k_fft_dx_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C1, Grid);
    cufftExecZ2D(cufft_plan_Z2D, Dev_Temp_C1, Dev_out);
}

// compute x-gradient on variable grid, needs Grid.sizeNfft + Grid.sizeN memory
void grad_y(TCudaGrid2D Grid, double *Dev_W, double *Dev_out, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D){
    cufftExecD2Z(cufft_plan_D2Z, Dev_W, Dev_Temp_C1);
    k_normalize_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Grid);

    k_fft_dy_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C1, Grid);
    cufftExecZ2D(cufft_plan_Z2D, Dev_Temp_C1, Dev_out);
}




// compute hermite with derivatives in fourier space, uniform helper function fitted for all grids to utilize only input temporary variable
// input has size (NX+1)/2*NY and output 4*NX*NY, output is therefore used as temporary variable
void fourier_hermite(TCudaGrid2D Grid, cufftDoubleComplex *Dev_In, double *Dev_Out, cufftHandle cufft_plan) {

	// reshift for transforming so that we have enough space for everything
	Dev_Out += Grid.N - 2*Grid.Nfft;

	// dy and dxdy derivates are stored in later parts of output array, we can therefore use the first half as a temporary variable
	// start with computing dy derivative
	k_fft_dy_h<<<Grid.fft_blocks, Grid.threadsPerBlock>>>(Dev_In, (cufftDoubleComplex*)Dev_Out, Grid);

	// compute dxdy afterwards, to combine backwards transformations
	k_fft_dx_h<<<Grid.fft_blocks, Grid.threadsPerBlock>>>((cufftDoubleComplex*)Dev_Out, (cufftDoubleComplex*)(Dev_Out) + Grid.Nfft, Grid);

	// backwards transformation, store dx in position 3/4 and dy in position 4/4
	cufftExecZ2D(cufft_plan, (cufftDoubleComplex*)(Dev_Out) + Grid.Nfft, Dev_Out + 2*Grid.N + 2*Grid.Nfft);
	cufftExecZ2D(cufft_plan, (cufftDoubleComplex*)Dev_Out, Dev_Out + Grid.N + 2*Grid.Nfft);

	// now compute dx derivative on itself and store it in the right place
	k_fft_dx_h<<<Grid.fft_blocks, Grid.threadsPerBlock>>>(Dev_In, (cufftDoubleComplex*)Dev_Out, Grid);
	cufftExecZ2D(cufft_plan, (cufftDoubleComplex*)Dev_Out, Dev_Out + 2*Grid.Nfft);// x-derivative of the vorticity in Fourier space

	// shift again just before final store
	Dev_Out += 2*Grid.Nfft - Grid.N;

	// at last, store normal values
	cufftExecZ2D(cufft_plan, Dev_In, Dev_Out);
}


/*******************************************************************
*		 Computation of Global conservation values				   *
*******************************************************************/
std::string compute_conservation_targets(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufft_plan_coarse_D2Z, cufftHandle cufft_plan_coarse_Z2D, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D,
		cufftDoubleComplex *Dev_Temp_C1)
{
	// check if we want to save at this time, combine all variables if so
	bool save_now = false;
	SaveComputational* save_comp = SettingsMain.getSaveComputational();
	for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
		// instants - distance to target is smaller than threshhold
		if (save_comp[i_save].is_instant && t_now - save_comp[i_save].time_start + dt*1e-5 < dt_now && t_now - save_comp[i_save].time_start + dt*1e-5 >= 0) {
			save_now = true;
		}
		// intervals - modulo to steps with safety-increased targets is smaller than step
		if (!save_comp[i_save].is_instant && save_comp[i_save].conv
			&& fmod(t_now - save_comp[i_save].time_start + dt*1e-5, save_comp[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_comp[i_save].time_start
			&& t_now - dt*1e-5 <= save_comp[i_save].time_end) {
			save_now = true;
		}
	}

	std::string message = "";
	if (save_now) {
		// compute mesure values
		double mesure[4];
		Compute_Energy(mesure[0], Dev_Psi, Grid_psi);
		Compute_Enstrophy(mesure[1], Dev_W_coarse, Grid_coarse);
		Compute_Palinstrophy(Grid_coarse, mesure[2], Dev_W_coarse, Dev_Temp_C1, cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D);

		// wmax
		thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(Dev_W_coarse);
		double w_max = thrust::reduce(w_ptr, w_ptr + Grid_coarse.N, 0.0, thrust::maximum<double>());
		double w_min = thrust::reduce(w_ptr, w_ptr + Grid_coarse.N, 0.0, thrust::minimum<double>());
		mesure[3] = std::max(w_max, -w_min);

		// save
		writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Mesure/Time_s");  // time vector for data
		writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Mesure/Energy");
		writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Mesure/Enstrophy");
		writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Mesure/Palinstrophy");
		writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Mesure/Max_vorticity");

		// construct message
		message = "Coarse Cons : Energ = " + to_str(mesure[0], 8)
				+    " \t Enstr = " + to_str(mesure[1], 8)
				+ " \t Palinstr = " + to_str(mesure[2], 8)
				+ " \t Wmax = " + to_str(mesure[3], 8);
	}

	return message;
}



/*******************************************************************
*		 Sample on a specific grid and save everything	           *
*	i know this became quite a beast in terms of input parameters
*******************************************************************/
std::string sample_compute_and_write(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		MapStack Map_Stack, MapStack Map_Stack_f, TCudaGrid2D* Grid_sample, TCudaGrid2D Grid_discrete,
		double *Host_sample, double *Dev_sample,
		cufftHandle* cufft_plan_sample_D2Z, cufftHandle* cufft_plan_sample_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *Host_forward_particles_pos, double *Dev_forward_particles_pos, int forward_particles_block, int forward_particles_thread,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f,
		double *bounds, double *W_initial_discrete) {

	// check if we want to save at this time, combine all variables if so
	std::string message = "";
	SaveSample* save_sample = SettingsMain.getSaveSample();
	double mesure[4];  // it's fine if we only output it last time, thats enough i guess
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		// check each save and execute it independent
		bool save_now = false;
		// instants - distance to target is smaller than threshhold
		if (save_sample[i_save].is_instant && t_now - save_sample[i_save].time_start + dt*1e-5 < dt_now && t_now - save_sample[i_save].time_start + dt*1e-5 >= 0) {
			save_now = true;
		}
		// intervals - modulo to steps with safety-increased targets is smaller than step
		if (!save_sample[i_save].is_instant
			&& fmod(t_now - save_sample[i_save].time_start + dt*1e-5, save_sample[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_sample[i_save].time_start
			&& t_now - dt*1e-5 <= save_sample[i_save].time_end) {
			save_now = true;
		}

		if (save_now) {
			std::string save_var = save_sample[i_save].var;  // extract save variables
			// forwards map to get it done
			if (SettingsMain.getForwardMap()) {
				// compute only if we actually want to save, elsewhise its a lot of computations for nothing
				if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos
						or SettingsMain.getForwardParticles()) {
					// apply mapstack to map or particle positions
					if (!SettingsMain.getForwardParticles()) {
						apply_map_stack(Grid_sample[i_save], Map_Stack_f, Dev_ChiX_f, Dev_ChiY_f, (cufftDoubleReal*)Dev_Temp_C1, 1);
					}
					else {
						apply_map_stack_points(Grid_sample[i_save], Map_Stack_f, Dev_ChiX_f, Dev_ChiY_f, (cufftDoubleReal*)Dev_Temp_C1, 1,
								Dev_forward_particles_pos, (cufftDoubleReal*)Dev_Temp_C1+2*Grid_sample[i_save].N,
								SettingsMain.getForwardParticlesNum(), forward_particles_block, forward_particles_thread);
					}

					// save map by copying and saving offsetted data
					if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
						writeTimeVariable(SettingsMain, "Map_ChiX_f_"+to_str(Grid_sample[i_save].NX),
								t_now, Host_sample, (cufftDoubleReal*)Dev_Temp_C1, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N, 2);
						writeTimeVariable(SettingsMain, "Map_ChiY_f_"+to_str(Grid_sample[i_save].NX),
								t_now, Host_sample, (cufftDoubleReal*)Dev_Temp_C1 + 1, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N, 2);
					}

					// save position of forwarded particles
					if (SettingsMain.getForwardParticles()) {
						writeTimeVariable(SettingsMain, "Particles_f_pos_"+to_str(Grid_sample[i_save].NX),
								t_now, Host_forward_particles_pos, (cufftDoubleReal*)Dev_Temp_C1+2*Grid_sample[i_save].N, 2*SettingsMain.getForwardParticlesNum()*sizeof(double), 2*SettingsMain.getForwardParticlesNum());
					}
				}

			}

			// compute map to initial condition through map stack
			apply_map_stack(Grid_sample[i_save], Map_Stack, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, -1);

			// save map by copying and saving offsetted data
			if (save_var.find("Map") != std::string::npos or save_var.find("Chi") != std::string::npos
					or save_var.find("Map_b") != std::string::npos or save_var.find("Chi_b") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Map_ChiX_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, (cufftDoubleReal*)Dev_Temp_C1, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N, 2);
				writeTimeVariable(SettingsMain, "Map_ChiY_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, (cufftDoubleReal*)Dev_Temp_C1 + 1, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N, 2);
			}

			// passive scalar theta - 1 to switch for passive scalar
			if (save_var.find("Scalar") != std::string::npos or save_var.find("Theta") != std::string::npos) {
				k_h_sample_from_init<<<Grid_sample[i_save].blocksPerGrid, Grid_sample[i_save].threadsPerBlock>>>(Dev_sample, (cufftDoubleReal*)Dev_Temp_C1,
						Grid_sample[i_save], Grid_discrete, 1, SettingsMain.getScalarNum(), W_initial_discrete, SettingsMain.getScalarDiscrete());

				writeTimeVariable(SettingsMain, "Scalar_Theta_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);

				thrust::device_ptr<double> scal_ptr = thrust::device_pointer_cast(Dev_sample);
				double scal_int = Grid_sample[i_save].h * Grid_sample[i_save].h * thrust::reduce(scal_ptr, scal_ptr + Grid_sample[i_save].N, 0.0, thrust::plus<double>());
				writeAppendToBinaryFile(1, &scal_int, SettingsMain, "/Monitoring_data/Mesure/Scalar_integral_"+ to_str(Grid_sample[i_save].NX));
			}

			// compute vorticity - 0 to switch for vorticity
			k_h_sample_from_init<<<Grid_sample[i_save].blocksPerGrid, Grid_sample[i_save].threadsPerBlock>>>(Dev_sample, (cufftDoubleReal*)Dev_Temp_C1,
					Grid_sample[i_save], Grid_discrete, 0, SettingsMain.getInitialConditionNum(), W_initial_discrete, SettingsMain.getInitialDiscrete());

			// save vorticity
			if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Vorticity_W_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);
			}

			// compute enstrophy and palinstrophy
			Compute_Enstrophy(mesure[1], Dev_sample, Grid_sample[i_save]);
			Compute_Palinstrophy(Grid_sample[i_save], mesure[2], Dev_sample, Dev_Temp_C1, cufft_plan_sample_D2Z[i_save], cufft_plan_sample_Z2D[i_save]);

			// compute wmax
			thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(Dev_sample);
			double w_max = thrust::reduce(w_ptr, w_ptr + Grid_sample[i_save].N, 0.0, thrust::maximum<double>());
			double w_min = thrust::reduce(w_ptr, w_ptr + Grid_sample[i_save].N, 0.0, thrust::minimum<double>());
			mesure[3] = std::max(w_max, -w_min);

			// compute laplacian of vorticity
			if (save_var.find("Laplacian_W") != std::string::npos) {
				laplacian(Grid_sample[i_save], Dev_sample, Dev_sample+Grid_sample[i_save].N, Dev_Temp_C1, cufft_plan_sample_D2Z[i_save], cufft_plan_sample_Z2D[i_save]);

				writeTimeVariable(SettingsMain, "Laplacian_W_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+Grid_sample[i_save].N, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);
			}

			// compute gradient of vorticity
			if (save_var.find("Grad_W") != std::string::npos) {
				grad_x(Grid_sample[i_save], Dev_sample, Dev_sample+Grid_sample[i_save].N, Dev_Temp_C1, cufft_plan_sample_D2Z[i_save], cufft_plan_sample_Z2D[i_save]);

				writeTimeVariable(SettingsMain, "GradX_W_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+Grid_sample[i_save].N, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);

				grad_y(Grid_sample[i_save], Dev_sample, Dev_sample+Grid_sample[i_save].N, Dev_Temp_C1, cufft_plan_sample_D2Z[i_save], cufft_plan_sample_Z2D[i_save]);

				writeTimeVariable(SettingsMain, "GradY_W_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+Grid_sample[i_save].N, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);
			}

			// reuse sampled vorticity to compute psi, take fourier_hermite reshifting into account
			long int shift = 2*Grid_sample[i_save].Nfft - Grid_sample[i_save].N;
			psi_upsampling(Grid_sample[i_save], Dev_sample, Dev_Temp_C1, Dev_sample+shift, cufft_plan_sample_D2Z[i_save], cufft_plan_sample_Z2D[i_save]);

			if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Stream_function_Psi_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+shift, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);
			}
			if (save_var.find("Stream_H") != std::string::npos or save_var.find("Psi_H") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Stream_function_Psi_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+shift, 4*Grid_sample[i_save].sizeNReal, 4*Grid_sample[i_save].N);
			}
			if (save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Velocity_UX_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+shift+1*Grid_sample[i_save].N, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);
				writeTimeVariable(SettingsMain, "Velocity_UY_"+to_str(Grid_sample[i_save].NX),
						t_now, Host_sample, Dev_sample+shift+2*Grid_sample[i_save].N, Grid_sample[i_save].sizeNReal, Grid_sample[i_save].N);
			}

			// compute energy
			Compute_Energy(mesure[0], Dev_sample+shift, Grid_sample[i_save]);

			// save conservation properties
			writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Mesure/Time_s_"+ to_str(Grid_sample[i_save].NX));  // time vector for data
			writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Mesure/Energy_"+ to_str(Grid_sample[i_save].NX));
			writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Mesure/Enstrophy_"+ to_str(Grid_sample[i_save].NX));
			writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Mesure/Palinstrophy_"+ to_str(Grid_sample[i_save].NX));
			writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Mesure/Max_vorticity_"+ to_str(Grid_sample[i_save].NX));

			// construct message
			message = "Sample Cons : Energ = " + to_str(mesure[0], 8)
					+    " \t Enstr = " + to_str(mesure[1], 8)
					+ " \t Palinstr = " + to_str(mesure[2], 8)
					+ " \t Wmax = " + to_str(mesure[3], 8);
		}
	}
	return message;
}



/*******************************************************************
*							   Zoom								   *
*			sample vorticity with mapstack at arbitrary frame
*******************************************************************/
void Zoom(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		MapStack Map_Stack, MapStack Map_Stack_f, TCudaGrid2D* Grid_zoom, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_discrete,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f,
		double *Dev_Temp, double *W_initial_discrete, double *psi,
		double *Host_particles_pos, double *Dev_particles_pos,
		double *Host_forward_particles_pos, double *Dev_forward_particles_pos, int forward_particles_block, int forward_particles_thread,
		double *Host_debug)
{
	// check if we want to save at this time, combine all variables if so
	std::string i_num = to_str(t_now); if (abs(t_now - T_MAX) < 1) i_num = "final";
	SaveZoom* save_zoom = SettingsMain.getSaveZoom();
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
		// check each save and execute it independent
		bool save_now = false;
		// instants - distance to target is smaller than threshhold
		if (save_zoom[i_save].is_instant && t_now - save_zoom[i_save].time_start + dt*1e-5 < dt_now && t_now - save_zoom[i_save].time_start + dt*1e-5 >= 0) {
			save_now = true;
		}
		// intervals - modulo to steps with safety-increased targets is smaller than step
		if (!save_zoom[i_save].is_instant
			&& fmod(t_now - save_zoom[i_save].time_start + dt*1e-5, save_zoom[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_zoom[i_save].time_start
			&& t_now - dt*1e-5 <= save_zoom[i_save].time_end) {
			save_now = true;
		}

		if (save_now) {
			// create folder
			std::string sub_folder_name = "/Zoom_data/Time_" + i_num;
			std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
			mkdir(folder_name_now.c_str(), 0777);

			std::string save_var = save_zoom[i_save].var;

			double x_min, x_max, y_min, y_max;

			double x_width = save_zoom[i_save].width_x/2.0;
			double y_width = save_zoom[i_save].width_y/2.0;

			std::cout << "Zoom var: " << save_zoom[i_save].var << "\n";
			std::cout << "Zoom rep: " << save_zoom[i_save].rep << "\n";
			std::cout << "Zoom rep fac: " << save_zoom[i_save].rep_fac << "\n";

			// do repetetive zooms
			for(int zoom_ctr = 0; zoom_ctr < save_zoom[i_save].rep; zoom_ctr++){
				// create new subfolder for current zoom
				sub_folder_name = "/Zoom_data/Time_" + i_num + "/Zoom_" + to_str(zoom_ctr);
				folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
				mkdir(folder_name_now.c_str(), 0777);

				// construct frame bounds for this zoom
				x_min = save_zoom[i_save].pos_x - x_width;
				x_max = save_zoom[i_save].pos_x + x_width;
				y_min = save_zoom[i_save].pos_y - y_width;
				y_max = save_zoom[i_save].pos_y + y_width;
				// safe bounds in array
				double bounds[4] = {x_min, x_max, y_min, y_max};

				TCudaGrid2D Grid_zoom_i(Grid_zoom[i_save].NX, Grid_zoom[i_save].NY, bounds);

				// compute forwards map for map stack of zoom window first, as it can be discarded afterwards
				if (SettingsMain.getForwardMap()) {
					// compute only if we actually want to save, elsewhise its a lot of computations for nothing
					if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos
							or SettingsMain.getForwardParticles()) {
						// apply mapstack to map or particle positions
						if (!SettingsMain.getForwardParticles()) {
							apply_map_stack(Grid_zoom_i, Map_Stack_f, Dev_ChiX_f, Dev_ChiY_f, Dev_Temp, 1);
						}
						else {
							apply_map_stack_points(Grid_zoom_i, Map_Stack_f, Dev_ChiX_f, Dev_ChiY_f, Dev_Temp, 1,
									Dev_forward_particles_pos, Dev_Temp+2*Grid_zoom_i.N,
									SettingsMain.getForwardParticlesNum(), forward_particles_block, forward_particles_thread);
						}

						// save map by copying and saving offsetted data
						if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
							cudaMemcpy2D(Host_debug, sizeof(double), Dev_Temp+Grid_zoom_i.N, sizeof(double)*2,
									sizeof(double), Grid_zoom_i.N, cudaMemcpyDeviceToHost);
							writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Map_ChiX");
							cudaMemcpy2D(Host_debug, sizeof(double), Dev_Temp+Grid_zoom_i.N+1, sizeof(double)*2,
									sizeof(double), Grid_zoom_i.N, cudaMemcpyDeviceToHost);
							writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Map_ChiY");
						}

						if (SettingsMain.getForwardParticles()) {
							// copy particles to host
							cudaMemcpy(Host_forward_particles_pos, Dev_Temp+2*Grid_zoom_i.N, 2*SettingsMain.getParticlesNum()*sizeof(double), cudaMemcpyDeviceToHost);

							int part_counter = 0;
							for (int i_p = 0; i_p < SettingsMain.getParticlesNum(); i_p++) {
								// check if particle in frame and then save it inside itself by checking for NaN values
								if (Host_forward_particles_pos[2*i_p] == Host_forward_particles_pos[2*i_p] and
									Host_forward_particles_pos[2*i_p+1] == Host_forward_particles_pos[2*i_p+1]) {
									// transcribe particle
									Host_forward_particles_pos[2*part_counter] = Host_forward_particles_pos[2*i_p];
									Host_forward_particles_pos[2*part_counter+1] = Host_forward_particles_pos[2*i_p+1];
									// increment counter
									part_counter++;
								}
							}
							// save particles
							writeAllRealToBinaryFile(2*part_counter, Host_forward_particles_pos, SettingsMain, sub_folder_name+"/Particles_pos_Fluid_f");
						}
					}
				}



				// compute backwards map for map stack of zoom window
				apply_map_stack(Grid_zoom_i, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_Temp+Grid_zoom_i.N, -1);

				// save map by copying and saving offsetted data
				if (save_var.find("Map") != std::string::npos or save_var.find("Chi") != std::string::npos) {
					cudaMemcpy2D(Host_debug, sizeof(double), Dev_Temp+Grid_zoom_i.N, sizeof(double)*2,
							sizeof(double), Grid_zoom_i.N, cudaMemcpyDeviceToHost);
					writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Map_ChiX");
					cudaMemcpy2D(Host_debug, sizeof(double), Dev_Temp+Grid_zoom_i.N+1, sizeof(double)*2,
							sizeof(double), Grid_zoom_i.N, cudaMemcpyDeviceToHost);
					writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Map_ChiY");
				}

				// passive scalar theta - 1 to switch for passive scalar
				if (save_var.find("Scalar") != std::string::npos or save_var.find("Theta") != std::string::npos) {
					k_h_sample_from_init<<<Grid_zoom_i.blocksPerGrid, Grid_zoom_i.threadsPerBlock>>>(Dev_Temp, Dev_Temp+Grid_zoom_i.N,
							Grid_zoom_i, Grid_discrete, 1, SettingsMain.getScalarNum(), W_initial_discrete, SettingsMain.getScalarDiscrete());

					cudaMemcpy(Host_debug, Dev_Temp, Grid_zoom_i.sizeNReal, cudaMemcpyDeviceToHost);
					writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Scalar_Theta");
				}

				// compute and save vorticity zoom
				if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos) {
					// compute vorticity - 0 to switch for vorticity
					k_h_sample_from_init<<<Grid_zoom_i.blocksPerGrid, Grid_zoom_i.threadsPerBlock>>>(Dev_Temp, Dev_Temp+Grid_zoom_i.N,
							Grid_zoom_i, Grid_discrete, 0, SettingsMain.getInitialConditionNum(), W_initial_discrete, SettingsMain.getInitialDiscrete());

					cudaMemcpy(Host_debug, Dev_Temp, Grid_zoom_i.sizeNReal, cudaMemcpyDeviceToHost);
					writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Vorticity_W");
				}

				// compute sample of stream function - it's not a zoom though!
				if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
					// sample stream function from hermite
					k_h_sample<<<Grid_zoom_i.blocksPerGrid,Grid_zoom_i.threadsPerBlock>>>(psi, Dev_Temp, Grid_psi, Grid_zoom_i);

					// save psi zoom
					cudaMemcpy(Host_debug, Dev_Temp, Grid_zoom_i.sizeNReal, cudaMemcpyDeviceToHost);
					writeAllRealToBinaryFile(Grid_zoom_i.N, Host_debug, SettingsMain, sub_folder_name+"/Stream_function_Psi");
				}

				// safe particles in zoomframe if wanted
				if (SettingsMain.getParticles() and (save_var.find("Particles") != std::string::npos or save_var.find("P") != std::string::npos)) {

					// copy particles to host
					cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*SettingsMain.getParticlesNum()*SettingsMain.getParticlesTauNum()*sizeof(double), cudaMemcpyDeviceToHost);

					// primitive loop on host, maybe this could be implemented more clever, but ca marche
					for (int i_tau = 0; i_tau < SettingsMain.getParticlesTauNum(); i_tau++) {
						int part_counter = 0;
						int tau_shift = 2*i_tau*SettingsMain.getParticlesNum();
						for (int i_p = 0; i_p < SettingsMain.getParticlesNum(); i_p++) {
							// check if particle in frame and then save it inside itself
							if (Host_particles_pos[2*i_p   + tau_shift] > x_min and
								Host_particles_pos[2*i_p   + tau_shift] < x_max and
								Host_particles_pos[2*i_p+1 + tau_shift] > y_min and
								Host_particles_pos[2*i_p+1 + tau_shift] < y_max) {
								// transcribe particle
								Host_particles_pos[2*part_counter   + tau_shift] = Host_particles_pos[2*i_p   + tau_shift];
								Host_particles_pos[2*part_counter+1 + tau_shift] = Host_particles_pos[2*i_p+1 + tau_shift];
								// increment counter
								part_counter++;
							}
						}
						// save particles
						string tau_name;
						if (i_tau == 0) tau_name = "Fluid"; else tau_name = "Tau=" + to_str(SettingsMain.particles_tau[i_tau]);
						writeAllRealToBinaryFile(2*part_counter, Host_particles_pos+tau_shift, SettingsMain, sub_folder_name+"/Particles_pos_"+ tau_name);
					}
				}

				x_width *=  save_zoom[i_save].rep_fac;
				y_width *=  save_zoom[i_save].rep_fac;
			}
		}
	}
}


// avoid overstepping specific time targets
double compute_next_timestep(SettingsCMM SettingsMain, double t, double dt) {
	double dt_now = dt;
	double dt_e = dt*1e-5;  // check for floating point arithmetic

	// 1st - particles
	double target_p = SettingsMain.getParticles()*SettingsMain.getParticlesSnapshotsPerSec();
	if (target_p > 0) {
		double target_now = 1.0/(double)target_p;
		// check for when modulo goes back to zero
		if(fmod(t+dt_e, target_now) > fmod(t + dt + dt_e, target_now)) {
			// fmod is still large, so we compute what is left to have a reminder of 0
			dt_now = fmin(dt_now, target_now - fmod(t, target_now));
		}
	}

	// 2nd - save_computational for instant and interval
	SaveComputational* save_comp = SettingsMain.getSaveComputational();
	for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
		// instants - distance to target goes from negative to positive
		if (save_comp[i_save].is_instant && t + dt_e - save_comp[i_save].time_start < 0 && t + dt_e + dt - save_comp[i_save].time_start > 0) {
			dt_now = fmin(dt_now, save_comp[i_save].time_start - t);
		}
		// intervals - modulo to steps with safety-increased targets is smaller than current timestep
		if (!save_comp[i_save].is_instant
			&& (fmod(t + dt_e - save_comp[i_save].time_start, save_comp[i_save].time_step) > fmod(t + dt + dt_e - save_comp[i_save].time_start, save_comp[i_save].time_step)
			|| fmod(t + dt_e - save_comp[i_save].time_start, save_comp[i_save].time_step) < 0 && fmod(t + dt + dt_e - save_comp[i_save].time_start, save_comp[i_save].time_step) > 0)
			&& t + dt + dt_e >= save_comp[i_save].time_start && t - dt_e <= save_comp[i_save].time_end) {
			dt_now = fmin(dt_now, save_comp[i_save].time_step - fmod(t, save_comp[i_save].time_step));
		}
	}

	// 3nd - save_sample for instant and interval
	SaveSample* save_sample = SettingsMain.getSaveSample();
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		// instants - distance to target goes from negative to positive
		if (save_sample[i_save].is_instant && t + dt_e - save_sample[i_save].time_start < 0 && t + dt_e + dt - save_sample[i_save].time_start > 0) {
			dt_now = fmin(dt_now, save_sample[i_save].time_start - t);
		}
		// intervals - modulo to steps with safety-increased targets is smaller than current timestep
		if (!save_sample[i_save].is_instant
			&& (fmod(t + dt_e - save_sample[i_save].time_start, save_sample[i_save].time_step) > fmod(t + dt + dt_e - save_sample[i_save].time_start, save_sample[i_save].time_step)
			|| fmod(t + dt_e - save_sample[i_save].time_start, save_sample[i_save].time_step) < 0 && fmod(t + dt + dt_e - save_sample[i_save].time_start, save_sample[i_save].time_step) > 0)
			&& t + dt + dt_e >= save_sample[i_save].time_start && t - dt_e <= save_sample[i_save].time_end) {
			dt_now = fmin(dt_now, save_sample[i_save].time_step - fmod(t, save_sample[i_save].time_step));
		}
	}

	// 4th - save_zoom for instant and interval
	SaveZoom* save_zoom = SettingsMain.getSaveZoom();
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
		// instants - distance to target goes from negative to positive
		if (save_zoom[i_save].is_instant && t + dt_e - save_zoom[i_save].time_start < 0 && t + dt_e + dt - save_zoom[i_save].time_start > 0) {
			dt_now = fmin(dt_now, save_zoom[i_save].time_start - t);
		}
		// intervals - modulo to steps with safety-increased targets is smaller than current timestep
		if (!save_zoom[i_save].is_instant
			&& (fmod(t + dt_e - save_zoom[i_save].time_start, save_zoom[i_save].time_step) > fmod(t + dt + dt_e - save_zoom[i_save].time_start, save_zoom[i_save].time_step)
			|| fmod(t + dt_e - save_zoom[i_save].time_start, save_zoom[i_save].time_step) < 0 && fmod(t + dt + dt_e - save_zoom[i_save].time_start, save_zoom[i_save].time_step) > 0)
			&& t + dt + dt_e >= save_zoom[i_save].time_start && t - dt_e <= save_zoom[i_save].time_end) {
			dt_now = fmin(dt_now, save_zoom[i_save].time_step - fmod(t, save_zoom[i_save].time_step));
		}
	}

	return dt_now;
}


/*******************************************************************
*				Zoom for a specific time instant				   *
*******************************************************************/

// We have to check that it still works.
/*
void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb){


	double LX;
	int NXc, NYc;
	int NXsf, NYsf;
	int map_stack_ctr = 23;									// don't need it, it can be tertemined by the size of data loaded...

	LX = twoPI;
	NXc = NYc = grid_scale;
	NXsf = NYsf = fine_grid_scale;

	string simulationName = File;

	TCudaGrid2D Gc(NXc, NYc, LX);
	TCudaGrid2D Gsf(NXsf, NYsf, LX);


	double *ChiX, *ChiY, *ChiX_stack, *ChiY_stack;
	ChiX = new double[4*grid_scale*grid_scale];
	ChiY = new double[4*grid_scale*grid_scale];
	ChiX_stack = new double[map_stack_ctr * 4*Grid_coarse.sizeNReal];
	ChiY_stack = new double[map_stack_ctr * 4*Grid_coarse.sizeNReal];


	readAllRealFromBinaryFile(4*grid_scale*grid_scale, ChiX, simulationName, "ChiX_" + t_nb);
	readAllRealFromBinaryFile(4*grid_scale*grid_scale, ChiY, simulationName, "ChiY_" + t_nb);
	readAllRealFromBinaryFile(map_stack_ctr * 4*grid_scale*grid_scale, ChiX_stack, simulationName, "ChiX_stack_" + t_nb);
	readAllRealFromBinaryFile(map_stack_ctr * 4*grid_scale*grid_scale, ChiY_stack, simulationName, "ChiY_stack_" + t_nb);


	double *Dev_W_fine;
	cudaMalloc((void**)&Dev_W_fine,  Grid_fine.sizeNReal);

	double *Dev_ChiX, *Dev_ChiY;
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);

	double *Dev_ChiX_stack, *Dev_ChiY_stack;
	cudaMalloc((void **) &Dev_ChiX_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal);
	cudaMalloc((void **) &Dev_ChiY_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal);


	cudaMemcpy(Dev_ChiX, ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiY, ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiX_stack, ChiX_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiY_stack, ChiY_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);


	Zoom(simulationName, LX, Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Dev_W_fine, map_stack_ctr);


	delete [] ChiX;
	delete [] ChiY;

	cudaFree(Dev_W_fine);
	cudaFree(Dev_ChiX);
	cudaFree(Dev_ChiY);
	cudaFree(Dev_ChiX_stack);
	cudaFree(Dev_ChiY_stack);


	printf("Finished\n");

}
*/
