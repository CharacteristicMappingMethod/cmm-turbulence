#ifndef __CUDA_PARTICLES_2D_H__
#define __CUDA_PARTICLES_2D_H__

#include "../numerical/cmm-hermite.h"
#include "../grid/cudagrid2d.h"


#ifdef __CUDACC__


__global__ void Rescale(int Nb_particle, double s, double *particles_pos);

__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi, int N, int NX, int NY, double h, int particles_time_integration_num);


__global__ void Particle_advect_inertia_init(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h);
__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h, double tau_p, int particles_time_integration_num);

//__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *Psi, double *Psi_previous, int N, int NX, int NY, double h, double tau_p);
//__global__ void Particle_advect_inertia_RK3(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *Psi, double *Psi_previous, double *Psi_previous_p, int N, int NX, int NY, double h, double tau_p);


#endif




#endif
