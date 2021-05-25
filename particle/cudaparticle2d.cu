#include "cudaparticle2d.h"
#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"

/*******************************************************************
*						  Particule advection        			   *
*******************************************************************/


__global__ void Rescale(int Nb_particle, ptype s, ptype *particles_pos){  // Rescaling the particles uniform distribution on the square [0,s]x[0,s] (s = twoPI)

  int i;
  int thread_finder = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
  for(i = pos; i < pos + step_pos; i++){

    particles_pos[2 * i] = s * particles_pos[2 * i];
    particles_pos[2 * i + 1] = s * particles_pos[2 * i + 1];
  }
}



// Advect particles using Hermite interpolation and RK2 for the time scheme.

__global__ void Particle_advect(int Nb_particle, ptype dt, ptype *particles_pos, ptype *psi, ptype *psi_previous, int N, int NX, int NY, ptype h){
  

  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  ptype psi_x, psi_y;
  ptype psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
  
  for(i = pos;i<pos + step_pos;i++){

    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
    device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);
    
    particles_pos[2 * i] = fmod(particles_pos[2 * i] +  0.5 * dt * (psi_y + psi_y_p) , twoPI);
    particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);
 
    
    particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1] -  0.5 * dt * (psi_x + psi_x_p), twoPI);
    particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);
    
  }
	
	
}


__global__ void Particle_advect_iner_ini(int Nb_particle, ptype dt, ptype *particles_pos, ptype *particles_vel, ptype *psi_previous, int N, int NX, int NY, ptype h){
  

  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  //ptype psi_x, psi_y;
  ptype psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
  
  for(i = pos;i<pos + step_pos;i++){

    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
    //device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

    particles_vel[2 * i] =  psi_y_p;
    particles_vel[2 * i + 1] =  -psi_x_p;
    
  }

}





__global__ void Particle_advect_inertia(int Nb_particle, ptype dt, ptype *particles_pos, ptype *particles_vel, ptype *psi_previous, int N, int NX, int NY, ptype h, ptype tau_p){
  

  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  //ptype psi_x, psi_y;
  ptype psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
  
  for(i = pos;i<pos + step_pos;i++){

    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
    //device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

    particles_vel[2 * i] =  particles_vel[2 * i] - dt * (particles_vel[2 * i] - psi_y_p) / tau_p;

    particles_pos[2 * i] = fmod(particles_pos[2 * i] + dt * particles_vel[2 * i], twoPI);
    particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);

    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - dt * (particles_vel[2 * i + 1] + psi_x_p) / tau_p;
    
    particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1], twoPI);
    particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);
    
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





