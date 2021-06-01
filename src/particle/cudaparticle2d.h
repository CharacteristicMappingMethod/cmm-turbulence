#ifndef __CUDA_PARTICLES_2D_H__
#define __CUDA_PARTICLES_2D_H__
#include "../grid/cudagrid2d.h"






	


#ifdef __CUDACC__
	

__global__ void Rescale(int Nb_particle, ptype s, ptype *particles_pos);

__global__ void Particle_advect(int Nb_particle, ptype dt, ptype *particles_pos, ptype *Psi, ptype *Psi_previous, int N, int NX, int NY, ptype h);

__global__ void Particle_advect_inertia(int Nb_particle, ptype dt, ptype *particles_pos, ptype *particles_vel, ptype *Psi, int N, int NX, int NY, ptype h, ptype tau_p);

__global__ void Particle_advect_iner_ini(int Nb_particle, ptype dt, ptype *particles_pos, ptype *particles_vel, ptype *Psi, int N, int NX, int NY, ptype h);	
	
	
	
#endif




#endif
