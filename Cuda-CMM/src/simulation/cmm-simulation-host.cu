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

#include "cmm-simulation-host.h"
#include "cmm-simulation-kernel.h"

#include "../numerical/cmm-hermite.h"
#include "../numerical/cmm-timestep.h"

#include "../numerical/cmm-fft.h"

#include "../ui/globals.h"

// debugging, using printf
#include "stdio.h"
#include <math.h>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "../numerical/cmm-mesure.h"


// function to get difference to 1 for thrust parallel reduction
struct absto1
{
    __host__ __device__
        double operator()(const double &x) const {
            return fabs(1-x);
        }
};
double incompressibility_check(TCudaGrid2D Grid_check, CmmVar2D ChiX, CmmVar2D ChiY, double *grad_Chi) {
	// compute determinant of gradient and save in gradchi
	k_incompressibility_check<<<Grid_check.blocksPerGrid, Grid_check.threadsPerBlock>>>(Grid_check, *ChiX.Grid, ChiX.Dev_var, ChiY.Dev_var, grad_Chi);
	// compute maximum using thrust parallel reduction
	thrust::device_ptr<double> grad_Chi_ptr = thrust::device_pointer_cast(grad_Chi);
	return thrust::transform_reduce(grad_Chi_ptr, grad_Chi_ptr + Grid_check.N, absto1(), 0.0, thrust::maximum<double>());
}

double invertibility_check(TCudaGrid2D Grid_check, CmmVar2D ChiX_b, CmmVar2D ChiY_b, CmmVar2D ChiX_f, CmmVar2D ChiY_f, double *abs_invert) {
	// compute determinant of gradient and save in gradchi
	k_invertibility_check<<<Grid_check.blocksPerGrid, Grid_check.threadsPerBlock>>>(Grid_check, *ChiX_b.Grid, *ChiX_f.Grid,
			ChiX_b.Dev_var, ChiY_b.Dev_var, ChiX_f.Dev_var, ChiY_f.Dev_var, abs_invert);

	// compute maximum using thrust parallel reduction
	thrust::device_ptr<double> abs_invert_ptr = thrust::device_pointer_cast(abs_invert);
	return thrust::reduce(abs_invert_ptr, abs_invert_ptr + Grid_check.N, 0.0, thrust::maximum<double>());
}


// wrapper function for map advection
void advect_using_stream_hermite(SettingsCMM SettingsMain, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Psi,
		double *Chi_new_X, double *Chi_new_Y, double *t, double *dt, int loop_ctr, int direction) {
	
	// compute lagrange coefficients from dt vector for timesteps n+dt and n+dt/2, this makes them dynamic
	double h_L1[4], h_L12[4];  // constant memory for lagrange coefficient to be computed only once
	int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;  // dt and t are shifted because of initial previous steps
	
	for (int i_p = 0; i_p < SettingsMain.getLagrangeOrder(); ++i_p) {
		h_L1[i_p] = get_L_coefficient(t, t[loop_ctr_l+1], loop_ctr_l, i_p, SettingsMain.getLagrangeOrder());
		h_L12[i_p] = get_L_coefficient(t, t[loop_ctr_l] + dt[loop_ctr_l+1]/2.0, loop_ctr_l, i_p, SettingsMain.getLagrangeOrder());
	}

	double h_c[3];  																		// constant memory for map update coefficient to be computed only once
	switch (SettingsMain.getMapUpdateOrderNum()) {
		case 2: { h_c[0] = +3.0/8.0; h_c[1] = -3.0/20.0; h_c[2] = +1.0/40.0; break; }  		// 6th order interpolation
		case 1: { h_c[0] = +1.0/3.0; h_c[1] = -1.0/12.0; break; }  							// 4th order interpolation
		case 0: { h_c[0] = +1.0/4.0; break; }  												// 2th order interpolation
	}

	double h_c1[12], h_cx[12], h_cy[12], h_cxy[12];  										// compute coefficients for each direction only once
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
	k_advect_using_stream_hermite<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var, Chi_new_X, Chi_new_Y,
			Psi.Dev_var, *ChiX.Grid, *Psi.Grid, t[loop_ctr_l+1], dt[loop_ctr_l+1],
			SettingsMain.getMapEpsilon(), SettingsMain.getTimeIntegrationNum(),
			SettingsMain.getMapUpdateOrderNum(), SettingsMain.getLagrangeOrder(), direction);
}



/*******************************************************************
*		 Apply mapstacks to get full map to initial condition	   *
*******************************************************************/
//void apply_map_stack(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction)
void apply_map_stack(TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY, double *Dev_Temp, int direction)
{
	// copy bounds to constant device memory
	cudaMemcpyToSymbol(d_bounds, Grid.bounds, sizeof(double)*4);

	// backwards map from last to first
	if (direction == -1) {
		// first application: current map
		k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);
		// afterwards: trace back all other maps
		for (int i_map = Map_Stack.map_stack_ctr-1; i_map >= 0; i_map--) {
			Map_Stack.copy_map_to_device(i_map);
			// printf("Map %d \n\n\n\n", i_map);
			k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
					Dev_Temp, *Map_Stack.Grid, Grid);
		}
	}
	// forward map from first to last
	else {
		// include remappings
		if (Map_Stack.map_stack_ctr > 0) {
			// first map to get map onto grid
			Map_Stack.copy_map_to_device(0);
			k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack, Dev_Temp, *Map_Stack.Grid, Grid);

			// loop over all other maps
			for (int i_map = 1; i_map < Map_Stack.map_stack_ctr; i_map++) {
				Map_Stack.copy_map_to_device(i_map);
				k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
						Dev_Temp, *Map_Stack.Grid, Grid);
			}

			// last map: current map
			k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);
		}
		// no remapping has occured yet
		else {
			k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);
		}
	}
}


/*******************************************************************
*		 Apply mapstacks to get full map to initial condition	   *
*		    and to map specific points / particles
*******************************************************************/
// this one is dirty and needs to be reworked
void apply_map_stack_points(SettingsCMM SettingsMain, TCudaGrid2D Grid, MapStack Map_Stack, double *ChiX, double *ChiY,
		double *Dev_Temp, int direction, std::map<std::string, CmmPart*> cmmPartMap, double *fluid_particles_pos_out)
{
	// copy bounds to constant device memory
	cudaMemcpyToSymbol(d_bounds, Grid.bounds, sizeof(double)*4);

	// backwards map from last to first
	if (direction == -1) {
		// first application: current map
		// sample map
		k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);

		// sample particles / specific points, do this for all particles and append them to output
		ParticlesForwarded *particles_forwarded = SettingsMain.getParticlesForwarded();
		double *pos_out_counter = fluid_particles_pos_out;
		for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
			std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
			CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
			if ((particles_forwarded[i_p].init_map != 0 && particles_forwarded[i_p].init_time != 0) || particles_forwarded[i_p].init_time == 0) {
				k_h_sample_points_map<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Map_Stack.Grid, Grid, ChiX, ChiY,
						Part_Pos_now.Dev_var, pos_out_counter, particles_forwarded[i_p].num);
			}
			// particles cannot be traced back, just give back the particle positions
			else {
				cudaMemcpy(pos_out_counter, Part_Pos_now.Dev_var, Part_Pos_now.sizeN, cudaMemcpyDeviceToDevice);
			}
			// shift pointer
			pos_out_counter += 2*particles_forwarded[i_p].num;
		}

		// afterwards: trace back all other maps
		for (int i_map = Map_Stack.map_stack_ctr-1; i_map >= 0; i_map--) {
			Map_Stack.copy_map_to_device(i_map);
			// sample map
			k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
					Dev_Temp, *Map_Stack.Grid, Grid);

			// sample particles / specific points
			double *pos_out_counter = fluid_particles_pos_out;
			pos_out_counter = fluid_particles_pos_out;
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
				CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
				// check if this map is applied too
				if ((i_map >= particles_forwarded[i_p].init_map && particles_forwarded[i_p].init_time != 0 && particles_forwarded[i_p].init_map != 0) || particles_forwarded[i_p].init_time == 0) {
					k_h_sample_points_map<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Map_Stack.Grid, Grid, ChiX, ChiY,
							pos_out_counter, pos_out_counter, particles_forwarded[i_p].num);
				}
				// shift pointer
				pos_out_counter += 2*particles_forwarded[i_p].num;
			}
		}
	}
	// forward map from first to last
	else {
		// include remappings
		if (Map_Stack.map_stack_ctr > 0) {
			// first map to get map onto grid
			Map_Stack.copy_map_to_device(0);
			// sample map
			k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack, Dev_Temp, *Map_Stack.Grid, Grid);
			// sample particles / specific points, do this for all particles and append them to output
			ParticlesForwarded *particles_forwarded = SettingsMain.getParticlesForwarded();
			double *pos_out_counter = fluid_particles_pos_out;
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
				CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
				// apply first map
				if (particles_forwarded[i_p].init_time == 0) {
					k_h_sample_points_map<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Map_Stack.Grid, Grid, Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
							Part_Pos_now.Dev_var, pos_out_counter, particles_forwarded[i_p].num);
				}
				// copy initially as we do not start at t=0
				else {
					cudaMemcpy(pos_out_counter, Part_Pos_now.Dev_var, Part_Pos_now.sizeN, cudaMemcpyDeviceToDevice);
				}
				// shift pointer
				pos_out_counter += 2*particles_forwarded[i_p].num;
			}

			// loop over all other maps
			for (int i_map = 1; i_map < Map_Stack.map_stack_ctr; i_map++) {
				Map_Stack.copy_map_to_device(i_map);
				k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
						Dev_Temp, *Map_Stack.Grid, Grid);

				pos_out_counter = fluid_particles_pos_out;
				for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
					std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
					CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
					// apply first map
					if ((i_map >= particles_forwarded[i_p].init_map && particles_forwarded[i_p].init_time != 0) || particles_forwarded[i_p].init_time == 0) {
						k_h_sample_points_map<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Map_Stack.Grid, Grid, Map_Stack.Dev_ChiX_stack, Map_Stack.Dev_ChiY_stack,
								pos_out_counter, pos_out_counter, particles_forwarded[i_p].num);
					}
					// shift pointer
					pos_out_counter += 2*particles_forwarded[i_p].num;
				}
			}

			// last map: current map
			k_apply_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);
			pos_out_counter = fluid_particles_pos_out;
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
				CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
				// apply first map
				if ((particles_forwarded[i_p].init_map != 0 && particles_forwarded[i_p].init_time != 0) || particles_forwarded[i_p].init_time == 0) {
					k_h_sample_points_map<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Map_Stack.Grid, Grid, ChiX, ChiY,
							pos_out_counter, pos_out_counter, particles_forwarded[i_p].num);
				}
				// shift pointer
				pos_out_counter += 2*particles_forwarded[i_p].num;
			}
		}
		// no remapping has occured yet
		else {
			// sample map
			k_h_sample_map_compact<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, *Map_Stack.Grid, Grid);
			// sample particles / specific points, do this for all particles and append them to output
			ParticlesForwarded *particles_forwarded = SettingsMain.getParticlesForwarded();
			double *pos_out_counter = fluid_particles_pos_out;
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
				CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
				k_h_sample_points_map<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Map_Stack.Grid, Grid, ChiX, ChiY,
						Part_Pos_now.Dev_var, pos_out_counter, particles_forwarded[i_p].num);

				pos_out_counter += 2*particles_forwarded[i_p].num;
			}
		}
	}
}


/*******************************************************************
*				Compute fine vorticity hermite					   *
*******************************************************************/

void translate_initial_condition_through_map_stack(MapStack Map_Stack, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Var,
		CmmVar2D Var_discrete_init, cufftDoubleComplex *Dev_Temp_C1, int simulation_num_c, bool initial_discrete, int var_num /*=0*/)
{
	/*
	@param Var: output variable to be translated through all maps
	@param Dev_Temp_C1: complex space temporary variable
	@param Var_discrete_init: initial condition as discrete array
	@param initial_discrete: if true, initial condition is used from discrete array
	@param simulation_num_c: number of simulation (inicond), used for continuous initial condition
	@param var_num: 0 vorticity (default), 1 passive scalar, 2 distribution function (optional), for continuous IC
	*/

	// Sample vorticity on fine grid
	// Var is used as temporary variable and output
	apply_map_stack(*Var.Grid, Map_Stack, ChiX.Dev_var, ChiY.Dev_var, Var.Dev_var+Var.Grid->N, -1);
	// initial condition from either discrete array or condition
	k_h_sample_from_init<<<Var.Grid->blocksPerGrid, Var.Grid->threadsPerBlock>>>(Var.Dev_var, Var.Dev_var+Var.Grid->N,
			*Var.Grid, *Var_discrete_init.Grid, var_num, simulation_num_c, Var_discrete_init.Dev_var, initial_discrete);

	// go to comlex space
	cufftExecD2Z(Var.plan_D2Z, Var.Dev_var, Dev_Temp_C1);
	k_normalize_h<<<Var.Grid->fft_blocks, Var.Grid->threadsPerBlock>>>(Dev_Temp_C1, *Var.Grid);

	// cut_off frequencies at N_fine/3 for turbulence (effectively 2/3)
//	k_fft_cut_off_scale<<<Grid_fineblocksPerGrid, Grid_finethreadsPerBlock>>>(Dev_Temp_C1, Grid_fineNX, (double)(Grid_fineNX)/3.0);

	// form hermite formulation
	fourier_hermite(*Var.Grid, Dev_Temp_C1, Var.Dev_var, Var.plan_Z2D);
}


/*******************************************************************
*						 Computation of Psi						   *
*******************************************************************/
void evaluate_stream_hermite(CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Vort_fine_init, CmmVar2D Psi, CmmVar2D empty_vort,
		cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi)
{
	/*
	This function computes the solution to $L \psi = w $ where $w$ is the vorticity and $\psi$ is the stream function
	@param Vort_fine_init: real space vorticity on fine grid as initial condition of sub-map
	@param Psi: real space stream function in Hermite form as output
	@param Dev_Temp_C1: complex space temporary variable
	@param empty_vort: this contains the grid and cufft_plans for the sampling size in real space, we then shift the grid in complex space
	*/


	// apply map to w and sample using mollifier, do it on a special grid for vorticity and apply mollification if wanted
	k_apply_map_and_sample_from_hermite<<<empty_vort.Grid->blocksPerGrid, empty_vort.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var,
			(cufftDoubleReal*)Dev_Temp_C1, Vort_fine_init.Dev_var, *ChiX.Grid, *empty_vort.Grid, *Vort_fine_init.Grid, molly_stencil, true);

	// forward fft, inline which is possible for forward fft
	cufftExecD2Z(empty_vort.plan_D2Z, (cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C1);
	k_normalize_h<<<empty_vort.Grid->fft_blocks, empty_vort.Grid->threadsPerBlock>>>(Dev_Temp_C1, *empty_vort.Grid);  // this is a normalization factor of FFT? if yes we dont need to do it everytime!!!

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3) and compute smooth W
	// use Psi grid here for intermediate storage
	//	k_fft_cut_off_scale<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_Temp_C1, Grid_coarse.NX, (double)(Grid_psi.NX)/3.0);

	// transition to stream function grid with three cases : grid_vort < grid_psi, grid_vort > grid_psi (a bit dumb) and grid_vort == grid_psi
	// grid change because inline data movement is nasty, we can use psi_real as buffer anyways
	if (empty_vort.Grid->NX != Psi.Grid->NX || empty_vort.Grid->NY != Psi.Grid->NY) {
		k_fft_grid_move<<<Psi.Grid->fft_blocks, Psi.Grid->threadsPerBlock>>>(Dev_Temp_C1, (cufftDoubleComplex*) Psi.Dev_var, *Psi.Grid, *empty_vort.Grid);
	}
	// no movement needed, just copy data over
	else {
		cudaMemcpy(Psi.Dev_var, Dev_Temp_C1, empty_vort.Grid->sizeNfft, cudaMemcpyDeviceToDevice);
	}

	// cut high frequencies in fourier space, however not that much happens after zero move add from coarse grid
	k_fft_cut_off_scale_h<<<Psi.Grid->fft_blocks, Psi.Grid->threadsPerBlock>>>((cufftDoubleComplex*) Psi.Dev_var, *Psi.Grid, freq_cut_psi);

	// Forming Psi hermite now on psi grid
	k_fft_iLap_h<<<Psi.Grid->fft_blocks, Psi.Grid->threadsPerBlock>>>((cufftDoubleComplex*) Psi.Dev_var, Dev_Temp_C1, *Psi.Grid);

	// Inverse laplacian in Fourier space
	fourier_hermite(*Psi.Grid, Dev_Temp_C1, Psi.Dev_var, Psi.plan_Z2D);
}
// debugging lines, could be needed here to check psi
//	cudaMemcpy(Host_Debug, Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(4*Grid_psi.N, Host_Debug, "psi_debug_4_nodes_C512_F2048_t64_T1", "Debug_2");


/*******************************************************************
*						 Computation of Psi						   *
*******************************************************************/

void evaluate_potential_from_density_hermite(SettingsCMM SettingsMain, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Vort_fine_init, CmmVar2D Psi,
		CmmVar2D empty_vort, cufftDoubleComplex *Dev_Temp_C1, int molly_stencil, double freq_cut_psi)
{	/*
	This function computes the solution to $L_xx \phi = (1-l \int f dv) $ where $w$ is the vorticity and $\phi$ is the stream function
	*/	
	// apply map to w and sample using mollifier, do it on a special grid for vorticity and apply mollification if wanted
	k_apply_map_and_sample_from_hermite<<<empty_vort.Grid->blocksPerGrid, empty_vort.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var,
			(cufftDoubleReal*)Dev_Temp_C1, Vort_fine_init.Dev_var, *ChiX.Grid, *empty_vort.Grid, *Vort_fine_init.Grid, molly_stencil, false);
	// this function solves the 1D laplace equation on the Grid_vort (coarse) and upsamples to Grid_Psi (fine)
	// writeTranferToBinaryFile(Grid_vort.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, "/f", false);
	get_psi_hermite_from_distribution_function(Psi, empty_vort, (cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C1);
//	get_psi_hermite_from_distribution_function(Psi_real, (cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C1, cufft_plan_phi_1D, cufft_plan_phi_1D_inverse,
//	cufft_plan_psi_D2Z, cufft_plan_psi_Z2D  ,Grid_vort, Grid_Psi);
	// writeTranferToBinaryFile(Grid_Psi.N, (cufftDoubleReal*)(Psi_real), SettingsMain, "/psi", false);
	// writeTranferToBinaryFile(Grid_Psi.N, (cufftDoubleReal*)(Psi_real+1*Grid_Psi.N), SettingsMain, "/dpsi_dx", false);
	// writeTranferToBinaryFile(Grid_Psi.N, (cufftDoubleReal*)(Psi_real+2*Grid_Psi.N), SettingsMain, "/dpsi_dy", false);
	// writeTranferToBinaryFile(Grid_Psi.N, (cufftDoubleReal*)(Psi_real+3*Grid_Psi.N), SettingsMain, "/dpsi_dxy", false);
   
// // 
}




/*
 * This function computes psi and the x-derivative of psi, we splice hermite function up for sampling, where only psi is needed
 * 		psi= phi  - v^2/2
 * 		d\psi/dx = dphi/dx = F^{-1} (ik_x F(phi))
 */
void get_psi_from_distribution_function(CmmVar2D Psi, CmmVar2D empty_vort, double *Dev_f_in, cufftDoubleComplex *Dev_Temp_C1)
{
	// #################################################################################################################
	// this function solves the 1D laplace equation on the Grid_vort (fine) and downsamples to Grid_Psi (between coarse and fine)
	// #################################################################################################################
	integral_v<<<empty_vort.Grid->blocksPerGrid.x, empty_vort.Grid->threadsPerBlock.x >>>((cufftDoubleReal*)Dev_f_in, (cufftDoubleReal*)Dev_Temp_C1+empty_vort.Grid->N, empty_vort.Grid->NX, empty_vort.Grid->NY, empty_vort.Grid->hy);
	// forward fft
	cufftExecD2Z (Psi.plan_1D_D2Z, (cufftDoubleReal*)Dev_Temp_C1+empty_vort.Grid->N, (cufftDoubleComplex*) Psi.Dev_var);
	// devide by NX to normalize FFT
	k_normalize_1D_h<<<empty_vort.Grid->fft_blocks.x, empty_vort.Grid->threadsPerBlock.x>>>((cufftDoubleComplex*) Psi.Dev_var, *empty_vort.Grid);  // this is a normalization factor of FFT? if yes we dont need to do it everytime!!!
	// inverse laplacian in fourier space - division by kx**2 and ky**2
	k_fft_iLap_h_1D<<<empty_vort.Grid->fft_blocks.x, empty_vort.Grid->threadsPerBlock.x>>>((cufftDoubleComplex*) Psi.Dev_var, (cufftDoubleComplex*)  Psi.Dev_var, *empty_vort.Grid);
	// do zero padding if needed
	if (empty_vort.Grid->NX != empty_vort.Grid->NX || empty_vort.Grid->NY != empty_vort.Grid->NY) {
		zero_padding_1D<<<empty_vort.Grid->fft_blocks.x, empty_vort.Grid->threadsPerBlock.x>>>((cufftDoubleComplex*)Psi.Dev_var, (cufftDoubleComplex*) Psi.Dev_var, *Psi.Grid, *empty_vort.Grid);
	}
	else { // no movement needed, just copy data over
		// cudaMemcpy(Psi_real_out, Dev_Temp_C1, Grid.sizeNfft, cudaMemcpyDeviceToDevice);
	}
	//compute x derivative for d\psi/dx = dphi/dx = F^{-1} (ik_x F(phi))
	k_fft_dx_h_1D<<<Psi.Grid->fft_blocks.x, Psi.Grid->threadsPerBlock.x>>>((cufftDoubleComplex*) Psi.Dev_var, (cufftDoubleComplex*) Dev_Temp_C1, *Psi.Grid);
	cufftExecZ2D (Psi.plan_1D_Z2D, (cufftDoubleComplex*) Dev_Temp_C1, (cufftDoubleReal*)(Psi.Dev_var+Psi.Grid->N));
	copy_first_row_to_all_rows_in_grid<<<Psi.Grid->blocksPerGrid, Psi.Grid->threadsPerBlock>>>((cufftDoubleReal*) (Psi.Dev_var+Psi.Grid->N), (cufftDoubleReal*) (Psi.Dev_var+Psi.Grid->N), *Psi.Grid);
	// inverse fft (1D)
	cufftExecZ2D (Psi.plan_1D_Z2D, (cufftDoubleComplex*) Psi.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1));
	// // assemble psi= phi  - v^2/2
	k_assemble_psi<<<Psi.Grid->blocksPerGrid, Psi.Grid->threadsPerBlock>>>((cufftDoubleReal*)(Dev_Temp_C1), Psi.Dev_var, *Psi.Grid);
}


/*
 * This function computes psi, x-, y- and xy-derivative of psi from distribution function
 * 		psi= phi  - v^2/2
 * 		d\psi/dx = dphi/dx = F^{-1} (ik_x F(phi))
 * 		d\psi/(dy) = v
 * 		d^2\psi/(dxdy) = 0
 */
void get_psi_hermite_from_distribution_function(CmmVar2D Psi, CmmVar2D empty_vort, double *Dev_f_in, cufftDoubleComplex *Dev_Temp_C1)
{
	// compute psi and d\psi/dx in external function
	get_psi_from_distribution_function(Psi, empty_vort, Dev_f_in, Dev_Temp_C1);

	// // compute y derivative for d\psi/(dy) = v
	generate_gridvalues_v<<<Psi.Grid->blocksPerGrid, Psi.Grid->threadsPerBlock>>>((cufftDoubleReal*)(Psi.Dev_var+2*Psi.Grid->N), -1.0, *Psi.Grid);
	// // compute xy derivative for d^2\psi/(dxdy) = 0

	cudaMemset(Psi.Dev_var+3*Psi.Grid->N, 0.0, Psi.Grid->sizeNReal);
//	set_value<<<Psi.Grid->blocksPerGrid, Psi.Grid->threadsPerBlock>>>((cufftDoubleReal*)(Psi.Dev_var+3*Psi.Grid->N), 0.0, *Psi.Grid);

	// cut high frequencies in fourier space, however not that much happens after zero move add from coarse grid
	// k_fft_cut_off_scale_h<<<Grid_Psi.fft_blocks, Grid_Psi.threadsPerBlock>>>((cufftDoubleComplex*) Dev_Temp_C1+Grid_Psi.Nfft, Grid_Psi, freq_cut_psi);
	// Inverse laplacian in Fourier space
	// fourier_hermite(Grid_psi, Dev_Temp_C1, Psi_real_out, cufft_plan_psi_Z2D);
}


/*******************************************************************
*		 			Main output function						   *
*		 streamlines all save process in one place				   *
*******************************************************************/
// this is again massive, maybe I can compress all that particle stuff
void save_functions(SettingsCMM& SettingsMain, Logger& logger, double t_now, double dt_now, double dt,
		MapStack Map_Stack, MapStack Map_Stack_f, std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap,
		cufftDoubleComplex *Dev_Temp_C1)
{
	std::string message;
	message = writeTimeStep(SettingsMain, t_now, dt_now, dt, cmmVarMap);
	if (SettingsMain.getVerbose() >= 3 && message != "") logger.push(message);

	// compute conservation if wanted
	message = compute_conservation_targets(SettingsMain, t_now, dt_now, dt, cmmVarMap, Dev_Temp_C1);
	if (SettingsMain.getVerbose() >= 3 && message != "") logger.push(message);

	// sample if wanted
	if (SettingsMain.getSimulationType() == "cmm_euler_2d") {
		message = sample_compute_and_write(SettingsMain, t_now, dt_now, dt, Map_Stack, Map_Stack_f, cmmVarMap, cmmPartMap, Dev_Temp_C1);
		if (SettingsMain.getVerbose() >= 3 && message != "") logger.push(message);
	}
	else if (SettingsMain.getSimulationType() == "cmm_vlasov_poisson_1d") {
		message = sample_compute_and_write_vlasov(SettingsMain, t_now, dt_now, dt, Map_Stack, Map_Stack_f, cmmVarMap, cmmPartMap, Dev_Temp_C1);
		if (SettingsMain.getVerbose() >= 3 && message != "") logger.push(message);
	}

	// save particle position if interested in that
	message = writeParticles(SettingsMain, t_now, dt_now, dt, cmmPartMap, *cmmVarMap["Psi"], (cufftDoubleReal*)Dev_Temp_C1);
	if (SettingsMain.getVerbose() >= 3 && message != "") logger.push(message);

	// zoom if wanted, has to be done after particle initialization, maybe a bit useless at first instance
	message = compute_zoom(SettingsMain, t_now, dt_now, dt, Map_Stack, Map_Stack_f, cmmVarMap, cmmPartMap, Dev_Temp_C1);
	if (SettingsMain.getVerbose() >= 3 && message != "") logger.push(message);
}


/*******************************************************************
*		 Computation of Global conservation values				   *
*******************************************************************/
std::string compute_conservation_targets(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		std::map<std::string, CmmVar2D*> cmmVarMap, cufftDoubleComplex *Dev_Temp_C1)
{
	// check if we want to save at this time, combine all variables if so
	bool save_now = false;
	SaveComputational* save_comp = SettingsMain.getSaveComputational();
	for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
		// instants - distance to target is smaller than threshhold
		if (save_comp[i_save].is_instant && t_now - save_comp[i_save].time_start + dt*1e-5 < dt_now && t_now - save_comp[i_save].time_start + dt*1e-5 >= 0 && save_comp[i_save].conv) {
			save_now = true;
		}
		// intervals - modulo to steps with safety-increased targets is smaller than step
		if (!save_comp[i_save].is_instant
			&& ((fmod(t_now - save_comp[i_save].time_start + dt*1e-5, save_comp[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_comp[i_save].time_start
			&& t_now - dt*1e-5 <= save_comp[i_save].time_end)
			|| t_now == save_comp[i_save].time_end)) {
			save_now = true;
		}
	}

	std::string message = "";
	if (save_now) {
		// compute mesure values
		double mesure[4];

		// compute quantities
		if (SettingsMain.getSimulationType() == "cmm_vlasov_poisson_1d"){
			// compute energies
			Compute_Kinetic_Energy(mesure[1], *cmmVarMap["Vort"], (cufftDoubleReal*) Dev_Temp_C1);
			Compute_Potential_Energy(mesure[2], *cmmVarMap["Psi"]);
			mesure[0]=mesure[1]+mesure[2]; // total energy
			// compute mass
			Compute_Mass(mesure[3], *cmmVarMap["Vort"]);

			// save
			writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Mesure/Time_s");  // time vector for data
			writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Mesure/Etot");
			writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Mesure/Ekin");
			writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Mesure/Epot");
			writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Mesure/Mass");

			// construct message
			message = "Computed coarse Cons : Etot = " + to_str(mesure[0], 8)
					+    " \t Ekin = " + to_str(mesure[1], 8)
					+ " \t Epot = " + to_str(mesure[2], 8)
					+ " \t Mass = " + to_str(mesure[3], 8);
		}
		else{
			Compute_Energy_H(mesure[0], *cmmVarMap["Psi"]);
//			Compute_Energy(mesure[0], *cmmVarMap["Psi"], Dev_Temp_C1);
			Compute_Enstrophy(mesure[1], *cmmVarMap["Vort"]);
			Compute_Palinstrophy(mesure[2], *cmmVarMap["Vort"], Dev_Temp_C1);
		
			// wmax
			thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(cmmVarMap["Vort"]->Dev_var);
			double w_max = thrust::reduce(w_ptr, w_ptr + cmmVarMap["Vort"]->Grid->N, 0.0, thrust::maximum<double>());
			double w_min = thrust::reduce(w_ptr, w_ptr + cmmVarMap["Vort"]->Grid->N, 0.0, thrust::minimum<double>());
			mesure[3] = std::max(w_max, -w_min);

			// hash of vorticity and stream function
			double w_hash[2]; Hash_array((char*)w_hash, cmmVarMap["Vort"]->Dev_var, cmmVarMap["Vort"]->Grid->N);
			double psi_hash[2]; Hash_array((char*)psi_hash, cmmVarMap["Psi"]->Dev_var, cmmVarMap["Psi"]->Grid->N);

			// save
			writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Mesure/Time_s");  // time vector for data
			writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Mesure/Energy");
			writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Mesure/Enstrophy");
			writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Mesure/Palinstrophy");
			writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Mesure/Max_vorticity");
			writeAppendToBinaryFile(2, w_hash, SettingsMain, "/Monitoring_data/Mesure/Hash_vorticity");
			writeAppendToBinaryFile(2, psi_hash, SettingsMain, "/Monitoring_data/Mesure/Hash_stream_function");

			// construct message
			message = "Computed coarse Cons : Energ = " + to_str(mesure[0], 8)
					+    " \t Enstr = " + to_str(mesure[1], 8)
					+ " \t Palinstr = " + to_str(mesure[2], 8)
					+ " \t Wmax = " + to_str(mesure[3], 8);
		}
	}

	return message;
}



/*******************************************************************
*		 Sample on a specific grid and save everything	           *
*	i know this became quite a beast in terms of input parameters
*******************************************************************/
std::string sample_compute_and_write(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, MapStack Map_Stack, MapStack Map_Stack_f,
		std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap, cufftDoubleComplex *Dev_Temp_C1)
{

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
			&& ((fmod(t_now - save_sample[i_save].time_start + dt*1e-5, save_sample[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_sample[i_save].time_start
			&& t_now - dt*1e-5 <= save_sample[i_save].time_end)
			|| t_now == save_sample[i_save].time_end)) {
			save_now = true;
		}

		if (save_now) {
			CmmVar2D *Sample_var = cmmVarMap["Sample_" + to_str(i_save)];  // extract sample variable
			std::string save_var = save_sample[i_save].var;  // extract save variables
			// forwards map to get it done
			if (SettingsMain.getForwardMap()) {
				// compute only if we actually want to save, elsewhise its a lot of computations for nothing
				if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos
						or SettingsMain.getParticlesForwardedNum() > 0) {
					// apply mapstack to map or particle positions
					if (SettingsMain.getParticlesForwardedNum() == 0) {
						apply_map_stack(*Sample_var->Grid, Map_Stack_f, cmmVarMap["ChiX_f"]->Dev_var, cmmVarMap["ChiY_f"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1, 1);
					}
					// forwarded particles: forward all particles regardless of if they will be saved, needs rework to be more clever
					else {
						apply_map_stack_points(SettingsMain, *Sample_var->Grid, Map_Stack_f, cmmVarMap["ChiX_f"]->Dev_var, cmmVarMap["ChiY_f"]->Dev_var,
								(cufftDoubleReal*)Dev_Temp_C1, 1, cmmPartMap, (cufftDoubleReal*)Dev_Temp_C1+2*Sample_var->Grid->N);
					}

					// save map by copying and saving offsetted data
					if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
						cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1, sizeof(double)*2,
												 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
						writeTimeVariable(SettingsMain, "Map_ChiX_f_"+to_str(Sample_var->Grid->NX),
								t_now, Sample_var->Dev_var, Sample_var->Grid->N);
						cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+1, sizeof(double)*2,
												 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
						writeTimeVariable(SettingsMain, "Map_ChiY_f_"+to_str(Sample_var->Grid->NX),
								t_now, Sample_var->Dev_var, Sample_var->Grid->N);
					}

					// save position of forwarded particles, go through all and only safe the needed ones
					double* particles_out_counter = (cufftDoubleReal*)Dev_Temp_C1+2*Sample_var->Grid->N;
					for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
						std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
						CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
						if (save_var.find("PartF_" + to_str_0(i_p+1, 2)) != std::string::npos) {
							// create particles folder if necessary
							std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
							std::string sub_folder_name = "/Particle_data/Time_" + t_s_now;
							std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
							struct stat st = {0};
							if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);
							// copy data to host and save
							writeTranferToBinaryFile(2*Part_Pos_now.num, particles_out_counter, SettingsMain, "/Particle_data/Time_" + t_s_now + "/Particles_forwarded_pos_P" + to_str_0(i_p+1, 2), 0);
						}
						particles_out_counter += 2*Part_Pos_now.num;  // increase counter
					}
				}

			}

			// compute map to initial condition through map stack
			apply_map_stack(*Sample_var->Grid, Map_Stack, cmmVarMap["ChiX"]->Dev_var, cmmVarMap["ChiY"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1, -1);

			// save map by copying and saving offsetted data
			if (save_var.find("Map_b") != std::string::npos or save_var.find("Chi_b") != std::string::npos) {
				cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1, sizeof(double)*2,
										 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
				writeTimeVariable(SettingsMain, "Map_ChiX_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
				cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+1, sizeof(double)*2,
										 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
				writeTimeVariable(SettingsMain, "Map_ChiY_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
			}

			// passive scalar theta - 1 to switch for passive scalar
			if (save_var.find("Scalar") != std::string::npos or save_var.find("Theta") != std::string::npos) {
				k_h_sample_from_init<<<Sample_var->Grid->blocksPerGrid, Sample_var->Grid->threadsPerBlock>>>(Sample_var->Dev_var, (cufftDoubleReal*)Dev_Temp_C1,
						*Sample_var->Grid, *cmmVarMap["Vort_discrete_init"]->Grid, 1, SettingsMain.getScalarNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getScalarDiscrete());

				writeTimeVariable(SettingsMain, "Scalar_Theta_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);

				thrust::device_ptr<double> scal_ptr = thrust::device_pointer_cast(Sample_var->Dev_var);
				double scal_int = Sample_var->Grid->hx * Sample_var->Grid->hy * thrust::reduce(scal_ptr, scal_ptr + Sample_var->Grid->N, 0.0, thrust::plus<double>());
				writeAppendToBinaryFile(1, &scal_int, SettingsMain, "/Monitoring_data/Mesure/Scalar_integral_"+ to_str(Sample_var->Grid->NX));
			}


			int varnum = 0;
			// compute vorticity - 0 to switch for vorticity
			k_h_sample_from_init<<<Sample_var->Grid->blocksPerGrid, Sample_var->Grid->threadsPerBlock>>>(Sample_var->Dev_var, (cufftDoubleReal*)Dev_Temp_C1,
					*Sample_var->Grid, *cmmVarMap["Vort_discrete_init"]->Grid, varnum, SettingsMain.getInitialConditionNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getInitialDiscrete());

			// save vorticity
			if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos  or save_var.find("F") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Vorticity_W_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
			}

			// compute enstrophy and palinstrophy
			Compute_Enstrophy(mesure[1], *Sample_var);
			Compute_Palinstrophy(mesure[2], *Sample_var, Dev_Temp_C1);

			// compute wmax
			thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(Sample_var->Dev_var);
			double w_max = thrust::reduce(w_ptr, w_ptr + Sample_var->Grid->N, 0.0, thrust::maximum<double>());
			double w_min = thrust::reduce(w_ptr, w_ptr + Sample_var->Grid->N, 0.0, thrust::minimum<double>());
			mesure[3] = std::max(w_max, -w_min);

			// hash of vorticity
			double w_hash[2]; Hash_array((char*)w_hash, Sample_var->Dev_var, Sample_var->Grid->N);

			// compute laplacian of vorticity
			if (save_var.find("Laplacian_W") != std::string::npos) {
				laplacian(*Sample_var, Sample_var->Dev_var+Sample_var->Grid->N, Dev_Temp_C1);

				writeTimeVariable(SettingsMain, "Laplacian_W_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var+Sample_var->Grid->N, Sample_var->Grid->N);
			}

			// compute gradient of vorticity
			if (save_var.find("Grad_W") != std::string::npos) {
				grad_x(*Sample_var, Sample_var->Dev_var+Sample_var->Grid->N, Dev_Temp_C1);

				writeTimeVariable(SettingsMain, "GradX_W_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var+Sample_var->Grid->N, Sample_var->Grid->N);

				grad_y(*Sample_var, Sample_var->Dev_var+Sample_var->Grid->N, Dev_Temp_C1);

				writeTimeVariable(SettingsMain, "GradY_W_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var+Sample_var->Grid->N, Sample_var->Grid->N);
			}

			// reuse sampled vorticity to compute psi, take fourier_hermite reshifting into account
			i_laplacian(*Sample_var, Sample_var->Dev_var, Dev_Temp_C1);
			Compute_Energy(mesure[0], *Sample_var, Dev_Temp_C1);

//			size_t shift = 2*Sample_var->Grid->Nfft - Sample_var->Grid->N;
//			i_laplacian_h(*Sample_var, Sample_var->Dev_var+shift, Dev_Temp_C1);
//			Sample_var->Dev_var = Sample_var->Dev_var+shift;
//
//			Compute_Energy_H(mesure[0], *Sample_var);

			// hash of stream function
			double psi_hash[2]; Hash_array((char*)psi_hash, Sample_var->Dev_var, Sample_var->Grid->N);

			if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Stream_function_Psi_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
			}
			// disable Hermite of stream function, saves space and all parts could be computed individually anyways
//			if (save_var.find("Stream_H") != std::string::npos or save_var.find("Psi_H") != std::string::npos) {
//				writeTimeVariable(SettingsMain, "Stream_function_Psi_"+to_str(Sample_var->Grid->NX),
//						t_now, Sample_var->Dev_var, 4*Sample_var->Grid->N);
//			}
			if (save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos) {
				grad_x(*Sample_var, Sample_var->Dev_var+Sample_var->Grid->N, Dev_Temp_C1);
				writeTimeVariable(SettingsMain, "Velocity_UX_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var+Sample_var->Grid->N, Sample_var->Grid->N);

				grad_y(*Sample_var, Sample_var->Dev_var+Sample_var->Grid->N, Dev_Temp_C1);

				writeTimeVariable(SettingsMain, "Velocity_UY_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var+Sample_var->Grid->N, Sample_var->Grid->N);
			}

			// shift back - only needed if computing hermite form
//			Sample_var->Dev_var = Sample_var->Dev_var-shift;

			// save conservation properties
			writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Mesure/Time_s_"+ to_str(Sample_var->Grid->NX));  // time vector for data
			writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Mesure/Energy_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Mesure/Enstrophy_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Mesure/Palinstrophy_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Mesure/Max_vorticity_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(2, w_hash, SettingsMain, "/Monitoring_data/Mesure/Hash_vorticity_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(2, psi_hash, SettingsMain, "/Monitoring_data/Mesure/Hash_stream_function_"+ to_str(Sample_var->Grid->NX));

			// construct message
			message = message + "Processed sample data " + to_str(i_save + 1) + " on grid " + to_str(Sample_var->Grid->NX) + ", Cons : Energ = " + to_str(mesure[0], 8)
					+    " \t Enstr = " + to_str(mesure[1], 8)
					+ " \t Palinstr = " + to_str(mesure[2], 8)
					+ " \t Wmax = " + to_str(mesure[3], 8);
		}
	}
	return message;
}




std::string sample_compute_and_write_vlasov(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, MapStack Map_Stack, MapStack Map_Stack_f,
		std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap, cufftDoubleComplex *Dev_Temp_C1)
{
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
			&& ((fmod(t_now - save_sample[i_save].time_start + dt*1e-5, save_sample[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_sample[i_save].time_start
			&& t_now - dt*1e-5 <= save_sample[i_save].time_end)
			|| t_now == save_sample[i_save].time_end)) {
			save_now = true;
		}

		if (save_now) {
			CmmVar2D *Sample_var = cmmVarMap["Sample_" + to_str(i_save)];  // extract sample variable
			std::string save_var = save_sample[i_save].var;  // extract save variables
			// forwards map to get it done
			if (SettingsMain.getForwardMap()) {
				// compute only if we actually want to save, elsewhise its a lot of computations for nothing
				if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos
						or SettingsMain.getParticlesForwardedNum() > 0) {
					// apply mapstack to map or particle positions
					if (SettingsMain.getParticlesForwardedNum() == 0) {
						apply_map_stack(*Sample_var->Grid, Map_Stack_f, cmmVarMap["ChiX_f"]->Dev_var, cmmVarMap["ChiY_f"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1, 1);
					}
					// forwarded particles: forward all particles regardless of if they will be saved, needs rework to be more clever
					else {
						apply_map_stack_points(SettingsMain, *Sample_var->Grid, Map_Stack_f, cmmVarMap["ChiX_f"]->Dev_var, cmmVarMap["ChiY_f"]->Dev_var,
								(cufftDoubleReal*)Dev_Temp_C1, 1, cmmPartMap, (cufftDoubleReal*)Dev_Temp_C1+2*Sample_var->Grid->N);
					}

					// save map by copying and saving offsetted data
					if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
						cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1, sizeof(double)*2,
												 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
						writeTimeVariable(SettingsMain, "Map_ChiX_f_"+to_str(Sample_var->Grid->NX),
								t_now, Sample_var->Dev_var, Sample_var->Grid->N);
						cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+1, sizeof(double)*2,
												 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
						writeTimeVariable(SettingsMain, "Map_ChiY_f_"+to_str(Sample_var->Grid->NX),
								t_now, Sample_var->Dev_var, Sample_var->Grid->N);
					}

					// save position of forwarded particles, go through all and only safe the needed ones
					double* particles_out_counter = (cufftDoubleReal*)Dev_Temp_C1+2*Sample_var->Grid->N;
					for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
						std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
						CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
						if (save_var.find("PartF_" + to_str_0(i_p+1, 2)) != std::string::npos) {
							// create particles folder if necessary
							std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
							std::string sub_folder_name = "/Particle_data/Time_" + t_s_now;
							std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
							struct stat st = {0};
							if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);
							// copy data to host and save
							writeTranferToBinaryFile(2*Part_Pos_now.num, particles_out_counter, SettingsMain, "/Particle_data/Time_" + t_s_now + "/Particles_forwarded_pos_P" + to_str_0(i_p+1, 2), 0);
						}
						particles_out_counter += 2*Part_Pos_now.num;  // increase counter
					}
				}

			}

			// compute map to initial condition through map stack
			apply_map_stack(*Sample_var->Grid, Map_Stack, cmmVarMap["ChiX"]->Dev_var, cmmVarMap["ChiY"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1, -1);

			// save map by copying and saving offsetted data
			if (save_var.find("Map_b") != std::string::npos or save_var.find("Chi_b") != std::string::npos) {
				cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1, sizeof(double)*2,
										 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
				writeTimeVariable(SettingsMain, "Map_ChiX_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
				cudaMemcpy2D(Sample_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+1, sizeof(double)*2,
										 sizeof(double), Sample_var->Grid->N, cudaMemcpyDeviceToDevice);
				writeTimeVariable(SettingsMain, "Map_ChiY_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
			}

			// passive scalar theta - 1 to switch for passive scalar
			if (save_var.find("Scalar") != std::string::npos or save_var.find("Theta") != std::string::npos) {
				k_h_sample_from_init<<<Sample_var->Grid->blocksPerGrid, Sample_var->Grid->threadsPerBlock>>>(Sample_var->Dev_var, (cufftDoubleReal*)Dev_Temp_C1,
						*Sample_var->Grid, *cmmVarMap["Vort_discrete_init"]->Grid, 1, SettingsMain.getScalarNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getScalarDiscrete());

				writeTimeVariable(SettingsMain, "Scalar_Theta_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);

				thrust::device_ptr<double> scal_ptr = thrust::device_pointer_cast(Sample_var->Dev_var);
				double scal_int = Sample_var->Grid->hx * Sample_var->Grid->hy * thrust::reduce(scal_ptr, scal_ptr + Sample_var->Grid->N, 0.0, thrust::plus<double>());
				writeAppendToBinaryFile(1, &scal_int, SettingsMain, "/Monitoring_data/Mesure/Scalar_integral_"+ to_str(Sample_var->Grid->NX));
			}


			int varnum = 2;
			// compute distribution function - 2 to switch for vlasov1D
			k_h_sample_from_init<<<Sample_var->Grid->blocksPerGrid, Sample_var->Grid->threadsPerBlock>>>(Sample_var->Dev_var, (cufftDoubleReal*)Dev_Temp_C1,
					*Sample_var->Grid, *cmmVarMap["Vort_discrete_init"]->Grid, varnum, SettingsMain.getInitialConditionNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getInitialDiscrete());

			// save particle distribution function
			if (save_var.find("Dist") != std::string::npos or save_var.find("F") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Distribution_F_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
			}

			// compute enstrophy and palinstrophy
			Compute_Kinetic_Energy(mesure[1], *Sample_var, (cufftDoubleReal*) Dev_Temp_C1);
			Compute_Mass(mesure[3], *Sample_var);

			// reuse sampled vorticity to compute psi,
//			get_psi_hermite_from_distribution_function(*Sample_var, *Sample_var, Sample_var->Dev_var, Dev_Temp_C1);

			// we only need psi and dx-derivative, so we do not need hermite function, this saves us space
			get_psi_from_distribution_function(*Sample_var, *Sample_var, Sample_var->Dev_var, Dev_Temp_C1);

			Compute_Potential_Energy(mesure[2], *Sample_var);

			if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
				writeTimeVariable(SettingsMain, "Stream_function_Psi_"+to_str(Sample_var->Grid->NX),
						t_now, Sample_var->Dev_var, Sample_var->Grid->N);
			}

			mesure[0] = mesure[1] + mesure[2]; // total energy
			// printf("DEBUG:    Etot = %.4f, Ekin = %.4f, Epot = %.4f, Mass = %.4f\n", mesure[0], mesure[1], mesure[2], mesure[3]);
			// error("babu", 1014);
			writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Mesure/Time_s_"+ to_str(Sample_var->Grid->NX));  // time vector for data
			writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Mesure/Etot_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Mesure/Ekin_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Mesure/Epot_"+ to_str(Sample_var->Grid->NX));
			writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Mesure/Mass_"+ to_str(Sample_var->Grid->NX));

				// construct message
			message = message + "Saved sample data " + to_str(i_save + 1) + " on grid " + to_str(Sample_var->Grid->NX) + ", Cons : Etot = " + to_str(mesure[0], 8)
					+    " \t Ekin = " + to_str(mesure[1], 8)
					+ " \t Epot = " + to_str(mesure[2], 8)
					+ " \t Mass = " + to_str(mesure[3], 8);
		}
	}
	return message;
}



/*******************************************************************
*							   Zoom								   *
*			sample vorticity with mapstack at arbitrary frame
*******************************************************************/
std::string compute_zoom(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, MapStack Map_Stack, MapStack Map_Stack_f,
		std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap, cufftDoubleComplex *Dev_Temp_C1)

//std::string compute_zoom(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
//		MapStack Map_Stack, MapStack Map_Stack_f, std::map<std::string, CmmVar2D*> cmmVarMap, cufftDoubleComplex *Dev_Temp_C1,
//		double **Host_particles_pos, double **Dev_particles_pos,
//		double **Host_forward_particles_pos, double **Dev_forward_particles_pos, int *forward_particles_block, int forward_particles_thread)
{
	// check if we want to save at this time, combine all variables if so
	std::string i_num = to_str(t_now); if (abs(t_now - T_MAX) < 1) i_num = "final";
	SaveZoom* save_zoom = SettingsMain.getSaveZoom();
	std::string message = "";
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
		// check each save and execute it independent
		bool save_now = false;
		// instants - distance to target is smaller than threshhold
		if (save_zoom[i_save].is_instant && t_now - save_zoom[i_save].time_start + dt*1e-5 < dt_now && t_now - save_zoom[i_save].time_start + dt*1e-5 >= 0) {
			save_now = true;
		}
		// intervals - modulo to steps with safety-increased targets is smaller than step
		if (!save_zoom[i_save].is_instant
			&& ((fmod(t_now - save_zoom[i_save].time_start + dt*1e-5, save_zoom[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_zoom[i_save].time_start
			&& t_now - dt*1e-5 <= save_zoom[i_save].time_end)
			|| t_now == save_zoom[i_save].time_end)) {
			save_now = true;
		}
		if (save_now) {
			CmmVar2D *Zoom_var = cmmVarMap["Zoom_" + to_str(i_save)];  // extract sample variable

			// create folder
			std::string sub_folder_name = "/Zoom_data/Time_" + i_num;
			std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
			mkdir(folder_name_now.c_str(), 0777);
			message = message + "Processed zoom data " + to_str(i_save + 1) + " on grid " + to_str(Zoom_var->Grid->NX);

			std::string save_var = save_zoom[i_save].var;

			double x_min, x_max, y_min, y_max;

			double x_width = save_zoom[i_save].width_x/2.0;
			double y_width = save_zoom[i_save].width_y/2.0;

			// do repetetive zooms
			for(int zoom_ctr = 0; zoom_ctr < save_zoom[i_save].rep; zoom_ctr++) {
				// create new subfolder for current zoom
				sub_folder_name = "/Zoom_data/Time_" + i_num + "/Zoom_" + to_str(i_save + 1) + "_rep_" + to_str(zoom_ctr);
				folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
				mkdir(folder_name_now.c_str(), 0777);

				// construct frame bounds for this zoom
				x_min = save_zoom[i_save].pos_x - x_width;
				x_max = save_zoom[i_save].pos_x + x_width;
				y_min = save_zoom[i_save].pos_y - y_width;
				y_max = save_zoom[i_save].pos_y + y_width;
				// safe bounds in array
				double bounds[4] = {x_min, x_max, y_min, y_max};

				printf("bounds : %f,  %f,  %f, %f\n", x_min, x_max, y_min, y_max);

				TCudaGrid2D Grid_zoom_i(Zoom_var->Grid->NX, Zoom_var->Grid->NY, Zoom_var->Grid->NZ, bounds);

				// compute forwards map for map stack of zoom window first, as it can be discarded afterwards
				if (SettingsMain.getForwardMap() and (save_var.find("Map_f") != std::string::npos
						or save_var.find("Chi_f") != std::string::npos or SettingsMain.getParticlesForwardedNum() > 0)) {
					
					// apply mapstack to map or particle positions
					if (SettingsMain.getParticlesForwardedNum() == 0) {
						apply_map_stack(Grid_zoom_i, Map_Stack_f, cmmVarMap["ChiX_f"]->Dev_var, cmmVarMap["ChiY_f"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1, 1);
					}
					// forwarded particles: forward all particles regardless of if they will be saved, needs rework to be more clever
					else {
						apply_map_stack_points(SettingsMain, Grid_zoom_i, Map_Stack_f, cmmVarMap["ChiX_f"]->Dev_var, cmmVarMap["ChiY_f"]->Dev_var,
								(cufftDoubleReal*)Dev_Temp_C1, 1, cmmPartMap, (cufftDoubleReal*)Dev_Temp_C1+2*Zoom_var->Grid->N);
					}

					// save map by copying and saving offsetted data
					if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
						cudaMemcpy2D(Zoom_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1, sizeof(double)*2,
												 sizeof(double), Zoom_var->Grid->N, cudaMemcpyDeviceToDevice);
						writeTimeVariable(SettingsMain, "Map_ChiX_f_"+to_str(Zoom_var->Grid->NX),
								t_now, Zoom_var->Dev_var, Zoom_var->Grid->N);
						cudaMemcpy2D(Zoom_var->Dev_var, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+1, sizeof(double)*2,
												 sizeof(double), Zoom_var->Grid->N, cudaMemcpyDeviceToDevice);
						writeTimeVariable(SettingsMain, "Map_ChiY_f_"+to_str(Zoom_var->Grid->NX),
								t_now, Zoom_var->Dev_var, Zoom_var->Grid->N);
					}

					// save position of forwarded particles, go through all and only safe the needed ones
					double* particles_out_counter = (cufftDoubleReal*)Dev_Temp_C1+2*Grid_zoom_i.N;
					for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
						std::string part_map_name = "PartF_Pos_P" + to_str_0(i_p+1, 2);
						CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
						if (save_var.find("PartF_" + to_str_0(i_p+1, 2)) != std::string::npos) {
							// create particles folder if necessary
							std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
							std::string sub_folder_name = "/Particle_data/Time_" + t_s_now;
							std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
							struct stat st = {0};
							if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);
							// copy data to host and save
							writeTranferToBinaryFile(2*Part_Pos_now.num, particles_out_counter, SettingsMain, "/Particle_data/Time_" + t_s_now + "/Particles_forwarded_pos_P" + to_str_0(i_p+1, 2), 0);
						}
						particles_out_counter += 2*Part_Pos_now.num;  // increase counter
					}
				}

				// compute backwards map for map stack of zoom window
				apply_map_stack(Grid_zoom_i, Map_Stack, cmmVarMap["ChiX"]->Dev_var, cmmVarMap["ChiY"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1+Grid_zoom_i.N, -1);

				// save map by copying and saving offsetted data
				if (save_var.find("Map_b") != std::string::npos or save_var.find("Chi_b") != std::string::npos) {
					cudaMemcpy2D((cufftDoubleReal*)Dev_Temp_C1, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+Grid_zoom_i.N, sizeof(double)*2,
							sizeof(double), Grid_zoom_i.N, cudaMemcpyDeviceToDevice);
					writeTranferToBinaryFile(Grid_zoom_i.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, sub_folder_name+"/Map_ChiX_"+to_str(Grid_zoom_i.NX), false);
					cudaMemcpy2D((cufftDoubleReal*)Dev_Temp_C1, sizeof(double), (cufftDoubleReal*)Dev_Temp_C1+Grid_zoom_i.N+1, sizeof(double)*2,
												sizeof(double), Grid_zoom_i.N, cudaMemcpyDeviceToDevice);
					writeTranferToBinaryFile(Grid_zoom_i.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, sub_folder_name+"/Map_ChiY_"+to_str(Grid_zoom_i.NX), false);
				}

				// passive scalar theta - 1 to switch for passive scalar
				if (save_var.find("Scalar") != std::string::npos or save_var.find("Theta") != std::string::npos) {
					k_h_sample_from_init<<<Grid_zoom_i.blocksPerGrid, Grid_zoom_i.threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1+Grid_zoom_i.N,
							Grid_zoom_i, *cmmVarMap["Vort_discrete_init"]->Grid, 1, SettingsMain.getScalarNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getScalarDiscrete());

					writeTranferToBinaryFile(Grid_zoom_i.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, sub_folder_name+"/Scalar_Theta_"+to_str(Grid_zoom_i.NX), false);
				}

				// compute and save vorticity zoom
				if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos) {
					// compute vorticity - 0 to switch for vorticity
					k_h_sample_from_init<<<Grid_zoom_i.blocksPerGrid, Grid_zoom_i.threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1+Grid_zoom_i.N,
							Grid_zoom_i, *cmmVarMap["Vort_discrete_init"]->Grid, 0, SettingsMain.getInitialConditionNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getInitialDiscrete());

					writeTranferToBinaryFile(Grid_zoom_i.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, sub_folder_name+"/Vorticity_W_"+to_str(Grid_zoom_i.NX), false);

					// hash
					double w_hash[2]; Hash_array((char*)w_hash, (cufftDoubleReal*)Dev_Temp_C1, Grid_zoom_i.N);
					writeAppendToBinaryFile(2, w_hash, SettingsMain, "/Monitoring_data/Mesure/Hash_vorticity_Zoom_" + to_str(i_save + 1) + "_rep_" + to_str(zoom_ctr));
				}

				if (save_var.find("Dist") != std::string::npos or save_var.find("F") != std::string::npos) {
					// compute distribution - 2 to switch for distribution
					k_h_sample_from_init<<<Grid_zoom_i.blocksPerGrid, Grid_zoom_i.threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1+Grid_zoom_i.N,
							Grid_zoom_i, *cmmVarMap["Vort_discrete_init"]->Grid, 2, SettingsMain.getInitialConditionNum(), cmmVarMap["Vort_discrete_init"]->Dev_var, SettingsMain.getInitialDiscrete());

					writeTranferToBinaryFile(Grid_zoom_i.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, sub_folder_name+"/Distribution_"+to_str(Grid_zoom_i.NX), false);
				}

				// compute sample of stream function - it's not a zoom though!
				if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
					// sample stream function from hermite
					k_h_sample<<<Grid_zoom_i.blocksPerGrid,Grid_zoom_i.threadsPerBlock>>>(cmmVarMap["Psi"]->Dev_var, (cufftDoubleReal*)Dev_Temp_C1, *cmmVarMap["Vort_discrete_init"]->Grid, Grid_zoom_i);

					// save psi zoom
					writeTranferToBinaryFile(Grid_zoom_i.N, (cufftDoubleReal*)Dev_Temp_C1, SettingsMain, sub_folder_name+"/Stream_function_Psi_"+to_str(Grid_zoom_i.NX), false);

					// hash
					double psi_hash[2]; Hash_array((char*)psi_hash, (cufftDoubleReal*)Dev_Temp_C1, Grid_zoom_i.N);
					writeAppendToBinaryFile(2, psi_hash, SettingsMain, "/Monitoring_data/Mesure/Hash_vorticity_Zoom_" + to_str(i_save + 1) + "_rep_" + to_str(zoom_ctr));

				}

				// safe particles in zoomframe if wanted
				// disabled for now, we can do this in postprocessing I think
//				ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
//				for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
//					// particle position first
//					if (save_var.find("PartA_" + to_str_0(i_p+1, 2)) != std::string::npos) {
//						// copy particles to host
//						cudaMemcpy(Host_particles_pos[i_p], Dev_particles_pos[i_p], 2*particles_advected[i_p].num*sizeof(double), cudaMemcpyDeviceToHost);
//						double* part_pos = Host_particles_pos[i_p];
//
//						int part_counter = 0;
//						for (int i_pn = 0; i_pn < particles_advected[i_pn].num; i_pn++) {
//							// check if particle in frame and then save it inside itself
//							if (part_pos[2*i_pn  ] > x_min and part_pos[2*i_pn  ] < x_max and
//								part_pos[2*i_pn+1] > y_min and part_pos[2*i_pn+1] < y_max) {
//								// transcribe particle
//								part_pos[2*part_counter  ] = part_pos[2*i_pn  ];
//								part_pos[2*part_counter+1] = part_pos[2*i_pn+1];
//								// increment counter
//								part_counter++;
//							}
//						}
//						// save particles
//						writeAllRealToBinaryFile(2*part_counter, Host_particles_pos[i_p], SettingsMain, sub_folder_name+"/Particles_advected_pos_P" + to_str_0(i_p, 2));
//					}
//				}

				x_width *=  save_zoom[i_save].rep_fac;
				y_width *=  save_zoom[i_save].rep_fac;
			}
		}
	}
	return message;
}


// avoid overstepping specific time targets
double compute_next_timestep(SettingsCMM SettingsMain, double t, double dt) {
	double dt_now = dt;
	double dt_e = dt*1e-5;  // check for floating point arithmetic

	// 1st - particles computation start positions for advected and forwarded particles
	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		if (t + dt_e - particles_advected[i_p].init_time < 0 && t + dt_e + dt - particles_advected[i_p].init_time > 0) {
			dt_now = fmin(dt_now, particles_advected[i_p].init_time - t);
		}
	}
	ParticlesForwarded* particles_forwarded = SettingsMain.getParticlesForwarded();
	for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
		if (t + dt_e - particles_forwarded[i_p].init_time < 0 && t + dt_e + dt - particles_forwarded[i_p].init_time > 0) {
			dt_now = fmin(dt_now, particles_forwarded[i_p].init_time - t);
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
