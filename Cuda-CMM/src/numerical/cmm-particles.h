#ifndef __CUDA_PARTICLES_H__
#define __CUDA_PARTICLES_H__

#include "../numerical/cmm-hermite.h"
#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

#ifdef __CUDACC__

	__global__ void k_rescale(int Nb_particle, double particles_center_x, double particles_center_y, double particles_width_x, double particles_width_y, double *particles_pos, double LX, double LY);


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
