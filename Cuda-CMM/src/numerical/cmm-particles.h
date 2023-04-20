/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
******************************************************************************************************************************/

#ifndef __CUDA_PARTICLES_H__
#define __CUDA_PARTICLES_H__

#include "../numerical/cmm-hermite.h"
#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

#ifdef __CUDACC__

	__global__ void k_rescale(int Nb_particle, double *particles_pos, TCudaGrid2D Grid);


	__global__ void Particle_advect_inertia_init(int Nb_particle, double *particles_pos, double *particles_vel,
			double *psi, TCudaGrid2D Grid);

	void particles_advect(SettingsCMM SettingsMain, TCudaGrid2D Grid_psi, double **particles_pos, double **particles_vel,
			double *psi, double *t, double *dt, int loop_ctr, int *particles_block, int particles_thread);
	void particles_advect(SettingsCMM SettingsMain, TCudaGrid2D Grid_psi, double **particles_pos, double **particles_vel,
			double *psi, double *t, double *dt, int loop_ctr, int* particles_block, int particles_thread, int i_p);

	__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi,
			TCudaGrid2D Grid, int particles_time_integration_num, int l_order);

	__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi,
			TCudaGrid2D Grid, double tau_p, int particles_time_integration_num, int l_order);

	//__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *Psi, double *Psi_previous, int N, int NX, int NY, double h, double tau_p);
	//__global__ void Particle_advect_inertia_RK3(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *Psi, double *Psi_previous, double *Psi_previous_p, int N, int NX, int NY, double h, double tau_p);

#endif

#endif
