#include "cmm-simulation-host.h"
#include "cmm-simulation-kernel.h"

#include "../numerical/cmm-hermite.h"
#include "../numerical/cmm-timestep.h"

#include "../numerical/cmm-fft.h"

#include "../ui/cmm-io.h"

// debugging, using printf
#include "stdio.h"

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
double incompressibility_check(double *ChiX, double *ChiY, double *gradChi, TCudaGrid2D Grid_check, TCudaGrid2D Grid_map) {
	// compute determinant of gradient and save in gradchi
	k_incompressibility_check<<<Grid_check.blocksPerGrid, Grid_check.threadsPerBlock>>>(ChiX, ChiY, gradChi, Grid_check, Grid_map);

	// compute maximum using thrust parallel reduction
	thrust::device_ptr<double> gradChi_ptr = thrust::device_pointer_cast(gradChi);
	return thrust::transform_reduce(gradChi_ptr, gradChi_ptr + Grid_check.N, absto1(), 0.0, thrust::maximum<double>());
}


// advect stream where footpoints are just neighbouring points
void advect_using_stream_hermite_grid(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, double *t, double *dt, int loop_ctr) {
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
			SettingsMain.getTimeIntegrationNum(), SettingsMain.getLagrangeOrder());

	// update map, x and y can be done seperately
	int shared_size = (18+2*SettingsMain.getMapUpdateOrderNum())*(18+2*SettingsMain.getMapUpdateOrderNum());  // how many points do we want to load?
	k_map_update<<<Grid_map.blocksPerGrid, Grid_map.threadsPerBlock, shared_size*sizeof(double)>>>(ChiX, Chi_new_X, Grid_map, SettingsMain.getMapUpdateOrderNum()+1, 0);
	k_map_update<<<Grid_map.blocksPerGrid, Grid_map.threadsPerBlock, shared_size*sizeof(double)>>>(ChiY, Chi_new_Y, Grid_map, SettingsMain.getMapUpdateOrderNum()+1, 1);
}


// wrapper function for map advection
void advect_using_stream_hermite(SettingsCMM SettingsMain, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double *ChiX, double *ChiY,
		double *Chi_new_X, double *Chi_new_Y, double *psi, double *t, double *dt, int loop_ctr) {
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
			SettingsMain.getMapUpdateOrderNum(), SettingsMain.getLagrangeOrder());
}



/*******************************************************************
*						 Apply mapstacks						   *
*******************************************************************/
void apply_map_stack_to_W(TCudaGrid2D Grid, TCudaGrid2D Grid_discrete, MapStack Map_Stack, double *ChiX, double *ChiY,
		double *W_real, double *Dev_Temp, double *W_initial_discrete, int simulation_num, bool initial_discrete)
{
	// copy bounds to constant device memory
	cudaMemcpyToSymbol(d_bounds, Grid.bounds, sizeof(double)*4);

	k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);

	// loop over all maps in map stack, where all maps are on host system
	// this could be parallelized between loading and computing?
	for (int i_map = Map_Stack.map_stack_ctr-1; i_map >= 0; i_map--) {
		Map_Stack.copy_map_to_device(i_map);
		k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
				Dev_Temp, *Map_Stack.Grid, Grid);
	}

	// initial condition
	k_h_sample_from_init<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(W_real, Dev_Temp, Grid, Grid_discrete, W_initial_discrete, simulation_num, initial_discrete);
}


/*******************************************************************
*				Compute fine vorticity hermite					   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_discrete, MapStack Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *W_initial_discrete, int simulation_num_c, bool initial_discrete)
{

	// Vorticity on coarse grid to vorticity on fine grid
	// W_H_real is used as temporary variable and output
	apply_map_stack_to_W(Grid_fine, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY,
			W_H_real, W_H_real+Grid_fine.N, W_initial_discrete, simulation_num_c, initial_discrete);

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
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real,
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

	// save vorticity on coarse grid, here Psi_real is used as buffer, assumption : Grid_psi > Grid_coarse
	k_fft_grid_move<<<Grid_coarse.fft_blocks, Grid_coarse.threadsPerBlock>>>(Dev_Temp_C1, (cufftDoubleComplex*) Psi_real, Grid_coarse, Grid_vort);
	cufftExecZ2D(cufft_plan_coarse_Z2D, (cufftDoubleComplex*) Psi_real, W_real);

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



// compute laplacian from vorticity, grid not set
void Laplacian_vort(TCudaGrid2D Grid, double *Dev_W, double *Dev_Lap, cufftDoubleComplex *Dev_Temp_C1, cufftHandle cufft_plan_D2Z, cufftHandle cufft_plan_Z2D){
    cufftExecD2Z(cufft_plan_D2Z, Dev_W, Dev_Temp_C1);
    k_normalize_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Grid);

    k_fft_lap_h<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C1, Grid);
    cufftExecZ2D(cufft_plan_Z2D, Dev_Temp_C1, Dev_Lap);
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
void compute_conservation_targets(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufft_plan_coarse_D2Z, cufftHandle cufft_plan_coarse_Z2D, cufftHandle cufftPlan_fine_D2Z, cufftHandle cufftPlan_fine_Z2D,
		cufftDoubleComplex *Dev_Temp_C1,
		double *Mesure, double *Mesure_fine, int count_mesure)
{
	// coarse grid
	Compute_Energy(&Mesure[3*count_mesure], Dev_Psi, Grid_psi);
	Compute_Enstrophy(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse);
	// fine grid, no energy because we do not have velocity on fine grid
	// fine vorticity disabled, as this needs another Grid_fine.sizeNReal in temp buffer, need to resolve later
//	Compute_Enstrophy(&Mesure_fine[1 + 3*count_mesure], Dev_W_H_fine, Grid_fine);

	// palinstrophy, fine first because vorticity is saved in temporal array
	// fine palinstrophy cannot be computed, as we have Dev_W_H_fine in Tev_Temp_C1, which is needed
//	Compute_Palinstrophy(Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_H_fine, Dev_Temp_C1, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D);
	Compute_Palinstrophy(Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Temp_C1, cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D);
}



/*******************************************************************
*		 Sample on a specific grid and save everything	           *
*******************************************************************/
void sample_compute_and_write(MapStack Map_Stack, TCudaGrid2D Grid_sample, TCudaGrid2D Grid_discrete, double *Host_sample, double *Dev_sample,
		cufftHandle cufft_plan_sample_D2Z, cufftHandle cufft_plan_sample_Z2D, cufftDoubleComplex *Dev_Temp_C1,
		double *Dev_ChiX, double*Dev_ChiY, double *bounds, double *W_initial_discrete, SettingsCMM SettingsMain, string i_num,
		double *Mesure_sample, int count_mesure) {

	// begin with vorticity
	apply_map_stack_to_W(Grid_sample, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY,
			Dev_sample, (cufftDoubleReal*)Dev_Temp_C1, W_initial_discrete, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete());
	writeTimeVariable(SettingsMain, "Vorticity_W_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample, Grid_sample.sizeNReal, Grid_sample.N);

	// compute enstrophy and palinstrophy
	Compute_Enstrophy(&Mesure_sample[1 + 3*count_mesure], Dev_sample, Grid_sample);
	Compute_Palinstrophy(Grid_sample, &Mesure_sample[2 + 3*count_mesure], Dev_sample, Dev_Temp_C1, cufft_plan_sample_D2Z, cufft_plan_sample_Z2D);

	// reuse sampled vorticity to compute psi, take fourier_hermite reshifting into account
	long int shift = 2*Grid_sample.Nfft - Grid_sample.N;
	psi_upsampling(Grid_sample, Dev_sample, Dev_Temp_C1, Dev_sample+shift, cufft_plan_sample_D2Z, cufft_plan_sample_Z2D);

	writeTimeVariable(SettingsMain, "Stream_function_Psi_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample+shift, 4*Grid_sample.sizeNReal, 4*Grid_sample.N);

	// compute energy
	Compute_Energy(&Mesure_sample[3*count_mesure], Dev_sample+shift, Grid_sample);

	// map
	k_h_sample_map<<<Grid_sample.blocksPerGrid,Grid_sample.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_sample, Dev_sample + Grid_sample.N, *Map_Stack.Grid, Grid_sample);
	writeTimeVariable(SettingsMain, "Map_ChiX_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample, Grid_sample.sizeNReal, Grid_sample.N);
	writeTimeVariable(SettingsMain, "Map_ChiY_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample + Grid_sample.N, Grid_sample.sizeNReal, Grid_sample.N);
}



/*******************************************************************
*							   Zoom								   *
*			sample vorticity with mapstack at arbitrary frame
*******************************************************************/
void Zoom(SettingsCMM SettingsMain, MapStack Map_Stack, TCudaGrid2D Grid_zoom, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_discrete,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_Temp, double *W_initial_discrete, double *psi,
		double *Host_particles_pos, double *Dev_particles_pos, double *Host_debug, string i_num)
{
	// create folder
	string sub_folder_name = "/Zoom_data/Time_" + i_num;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	mkdir(folder_name_now.c_str(), 0700);

	double x_min, x_max, y_min, y_max;

	double x_width = SettingsMain.getZoomWidthX()/2.0;
	double y_width = SettingsMain.getZoomWidthY()/2.0;

	// do repetetive zooms
	for(int zoom_ctr = 0; zoom_ctr < SettingsMain.getZoomRepetitions(); zoom_ctr++){
		// create new subfolder for current zoom
		sub_folder_name = "/Zoom_data/Time_" + i_num + "/Zoom_" + to_str(zoom_ctr);
		folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
		mkdir(folder_name_now.c_str(), 0700);

		// construct frame bounds for this zoom
		x_min = SettingsMain.getZoomCenterX() - x_width;
		x_max = SettingsMain.getZoomCenterX() + x_width;
		y_min = SettingsMain.getZoomCenterY() - y_width;
		y_max = SettingsMain.getZoomCenterY() + y_width;
		// safe bounds in array
		double bounds[4] = {x_min, x_max, y_min, y_max};

		TCudaGrid2D Grid_zoom_i(Grid_zoom.NX, Grid_zoom.NY, bounds);

		// compute zoom of vorticity
		apply_map_stack_to_W(Grid_zoom_i, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY,
				Dev_Temp, Dev_Temp+Grid_zoom_i.N, W_initial_discrete,
				SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete());

		// save vorticity zoom
		cudaMemcpy(Host_debug, Dev_Temp, Grid_zoom.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_zoom.N, Host_debug, SettingsMain, sub_folder_name+"/Vorticity_W");

		// compute sample of stream function - it's not a zoom though!
		if (SettingsMain.getZoomSavePsi()) {
			// sample stream funciton from hermite
			k_h_sample<<<Grid_zoom_i.blocksPerGrid,Grid_zoom_i.threadsPerBlock>>>(psi, Dev_Temp, Grid_psi, Grid_zoom_i);

			// save psi zoom
			cudaMemcpy(Host_debug, Dev_Temp, 4*Grid_zoom.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(4*Grid_zoom.N, Host_debug, SettingsMain, sub_folder_name+"/Stream_function_Psi");
		}

		// safe particles in zoomframe if wanted
		if (SettingsMain.getParticles() && SettingsMain.getZoomSaveParticles()) {

			// copy particles to host
		    cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*SettingsMain.getParticlesNum()*SettingsMain.getParticlesTauNum()*sizeof(double), cudaMemcpyDeviceToHost);

			// primitive loop on host, maybe this could be implemented more clever, but ca marche
			for (int i_tau = 0; i_tau < SettingsMain.getParticlesTauNum(); i_tau++) {
				int part_counter = 0;
				int tau_shift = 2*i_tau*SettingsMain.getParticlesNum();
				for (int i_p = 0; i_p < SettingsMain.getParticlesNum(); i_p++) {
					// check if particle in frame and then save it inside itself
					if (Host_particles_pos[2*i_p   + tau_shift] > x_min &&
						Host_particles_pos[2*i_p   + tau_shift] < x_max &&
						Host_particles_pos[2*i_p+1 + tau_shift] > y_min &&
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

		x_width *=  SettingsMain.getZoomRepetitionsFactor();
		y_width *=  SettingsMain.getZoomRepetitionsFactor();
	}
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
