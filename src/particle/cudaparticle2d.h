#ifndef __CUDA_PARTICLES_2D_H__
#define __CUDA_PARTICLES_2D_H__
#include "../grid/cudagrid2d.h"






	


#ifdef __CUDACC__
	
	
	__global__ void Particle_advect(int Nb_particle, ptype dt, ptype *particles_pos, ptype *Psy, int N, int NX, int NY);	
	
	
	
#endif




#endif
