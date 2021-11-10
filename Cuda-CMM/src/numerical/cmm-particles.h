#ifndef __CUDA_PARTICLES_H__
#define __CUDA_PARTICLES_H__

#include "../numerical/cmm-hermite.h"
#include "../grid/cudagrid2d.h"
#include "../simulation/settings.h"

#ifdef __CUDACC__

	__global__ void Rescale(int Nb_particle, double s, double *particles_pos);


	__global__ void Particle_advect_inertia_init(int Nb_particle, double dt, double *particles_pos, double *particles_vel,
			double *psi, TCudaGrid2D Grid);

	void particles_advect(SettingsCMM SettingsMain, TCudaGrid2D Grid_psi, double *particles_pos, double *particles_vel,
			double *psi, double *t, double *dt, int loop_ctr, int particles_block, int particles_thread);

	__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi,
			TCudaGrid2D Grid, int particles_time_integration_num, int l_order);

	__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi,
			TCudaGrid2D Grid, double tau_p, int particles_time_integration_num, int l_order);

	//__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *Psi, double *Psi_previous, int N, int NX, int NY, double h, double tau_p);
	//__global__ void Particle_advect_inertia_RK3(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *Psi, double *Psi_previous, double *Psi_previous_p, int N, int NX, int NY, double h, double tau_p);

#endif

#endif
