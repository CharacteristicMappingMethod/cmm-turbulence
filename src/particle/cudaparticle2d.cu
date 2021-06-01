#include "cudaparticle2d.h"


/*******************************************************************
*						  Particule advection        			   *
*******************************************************************/


__global__ void Particle_advect(int Nb_particle, ptype dt, ptype *particles_pos, ptype *psi, int N, int NX, int NY){
    
	int In; 
	int thread_finder = threadIdx.x + blockDim.x * blockIdx.x;
	int Di = blockDim.x * gridDim.x;
	
	int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
	
	for(int i=pos;i<pos + step_pos;i++){
		In = floor(particles_pos[2*i + 1] * NY) * NX + floor(particles_pos[2*i] * NX);
		particles_pos[2*i	 ] 	= fmod(particles_pos[2*i    ] + dt * psi[2*N + In], 1.);
		particles_pos[2*i	 ]	= (particles_pos[2*i] < 0)*(1+particles_pos[2*i]) + (particles_pos[2*i] > 0)*(particles_pos[2*i]);
		particles_pos[2*i + 1] 	= fmod(particles_pos[2*i + 1] - dt * psi[  N + In], 1.);
		particles_pos[2*i + 1]	= (particles_pos[2*i + 1] < 0)*(1+particles_pos[2*i + 1]) + (particles_pos[2*i + 1] > 0)*(particles_pos[2*i + 1]);
	}
	
	
	
	
	
}














/*
__global__ void Particle_advect(int Nb_particle, ptype dt, ptype *particles_pos, ptype *psi, int N, int NX, int NY){
	int In; 
	
	for(int i = 0; i < Nb_particle; i+=1){
		In = floor(particles_pos[2*i + 1] * NY) * NX + floor(particles_pos[2*i] * NX);
		particles_pos[2*i	 ] 	= fmod(particles_pos[2*i    ] + dt * psi[2*N + In], 1.);
		particles_pos[2*i	 ]	= (particles_pos[2*i] < 0)*(1+particles_pos[2*i]) + (particles_pos[2*i] > 0)*(particles_pos[2*i]);
		particles_pos[2*i + 1] 	= fmod(particles_pos[2*i + 1] - dt * psi[  N + In], 1.);
		particles_pos[2*i + 1]	= (particles_pos[2*i + 1] < 0)*(1+particles_pos[2*i + 1]) + (particles_pos[2*i + 1] > 0)*(particles_pos[2*i + 1]);
	}
}
*/





