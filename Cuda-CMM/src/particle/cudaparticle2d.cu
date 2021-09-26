#include "cudaparticle2d.h"
#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"



/*******************************************************************
*						  Particule advection        			   *
*******************************************************************/


__global__ void Rescale(int Nb_particle, double s, double *particles_pos){  // Rescaling the particles uniform distribution on the square [0,s]x[0,s] (s = twoPI)

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

__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi, double *psi_previous, double *psi_previous_p, int N, int NX, int NY, double h){
  

  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  double psi_x, psi_y;
  double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
  //double psi_x_p_p, psi_y_p_p;
  #ifdef RKThree_PARTICLES
      double psi_x_pp, psi_y_pp; // Previous values of the x and y derivative of Psi.
      double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
      double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;
      double l[3] = {0.375,0.75,-0.125};
  #endif
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;


  for(i = pos;i<pos + step_pos;i++){

    #ifndef RKThree_PARTICLES
        device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
        device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);
    
        particles_pos[2 * i] = fmod(particles_pos[2 * i] +  0.5 * dt * (psi_y + psi_y_p) , twoPI);
        particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);
 
    
        particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1] -  0.5 * dt * (psi_x + psi_x_p), twoPI);
        particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);

    #else

        // Compute k1
        device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
        u_p = psi_y_p;
        v_p = -psi_x_p;

        k1_x = u_p;
        k1_y = v_p;

        // Compute k2
        device_hermite_interpolate_dx_dy(psi_previous_p, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
        u_pp = psi_y_pp;
        v_pp = -psi_x_pp;

        device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
        u_p = psi_y_p;
        v_p = -psi_x_p;

        device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
        u = psi_y;
        v = -psi_x;

        k2_x = l[0] * u + l[1]* u_p + l[2] * u_pp;
        k2_y = l[0] * v + l[1] * v_p + l[2] * v_pp;


        // Compute k3

        device_hermite_interpolate_dx_dy(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
        u = psi_y;
        v = -psi_x;

        k3_x = u;
        k3_y = v;


        particles_pos[2 * i] = fmod(particles_pos[2 * i] +  dt * (k1_x + 4*k2_x + k3_x)/6 , twoPI);
        particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);


        particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1] + dt * (k1_y + 4*k2_y + k3_y)/6, twoPI);
        particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);

    #endif

  }
}


__global__ void Particle_advect_iner_ini(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi_previous, int N, int NX, int NY, double h){
  

  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  //double psi_x, psi_y;
  double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
  
  for(i = pos;i<pos + step_pos;i++){

    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
    //device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

    particles_vel[2 * i] =  psi_y_p;
    particles_vel[2 * i + 1] =  -psi_x_p;
    
  }
}





__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, double *psi_previous, int N, int NX, int NY, double h, double tau_p){
  
  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  //double psi_x, psi_y;
  double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
  
  for(i = pos;i<pos + step_pos;i++){

    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
    //device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

    //particles_vel[2 * i] =  particles_vel[2 * i] - dt * (particles_vel[2 * i] - psi_y_p) / tau_p;
    particles_vel[2 * i] =  particles_vel[2 * i] - 0.5*(dt/tau_p) * ((particles_vel[2 * i] - psi_y_p) + (particles_vel[2*i] + 0.5*dt*(particles_vel[2 * i] - psi_y_p)  - psi_y_p));

    particles_pos[2 * i] = fmod(particles_pos[2 * i] + dt * particles_vel[2 * i], twoPI);
    particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);

    //particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - dt * (particles_vel[2 * i + 1] + psi_x_p) / tau_p;
    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - 0.5*(dt/tau_p) * (particles_vel[2 * i + 1] + psi_x_p + (particles_vel[2*i + 1 ] + 0.5*dt*(particles_vel[2 * i] + psi_x_p) + psi_x_p));

    particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1], twoPI);
    particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);


  }
}


__global__ void Particle_advect_inertia_RK3(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, double *psi_previous, double *psi_previous_p, int N, int NX, int NY, double h, double tau_p){

  int i;
  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
  int Di = blockDim.x * gridDim.x;
  //double psi_x, psi_y;
  double psi_x, psi_y, psi_x_p, psi_y_p, psi_x_pp, psi_y_pp; // Previous values of the x and y derivative of Psi.
  double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
  double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;
  //double vix, viy;
  double l[3] = {0.375,0.75,-0.125};

  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;

  for(i = pos;i<pos + step_pos;i++){

    // Compute k1
    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
    u_p = psi_y_p;
    v_p = -psi_x_p;

    k1_x = particles_vel[2*i] - u_p;
    k1_y = particles_vel[2*i+1] - v_p;

    // Compute k2
    device_hermite_interpolate_dx_dy(psi_previous_p, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
    u_pp = psi_y_pp;
    v_pp = -psi_x_pp;

    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
    u_p = psi_y_p;
    v_p = -psi_x_p;

    device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
    u = psi_y;
    v = -psi_x;

    //k2_x = L1(t+dt/2, t, t-dt, t-2*dt) * u + L2(t+dt/2, t, t-dt, t-2*dt)* u_p + L3(t+dt/2, t, t-dt, t-2*dt)*u_p_p;
    //k2_y = L1(t+dt/2, t, t-dt, t-2*dt) * v + L2(t+dt/2, t, t-dt, t-2*dt)* v_p + L3(t+dt/2, t, t-dt, t-2*dt)*v_p_p;

    //vix = l[0] * particles_vel[2*i] + l[1]* particles_vel_previous[2*i] + l[2] * particles_vel_previous_p[2*i];
    //viy = l[0] * particles_vel[2*i + 1] + l[1]* particles_vel_previous[2*i + 1 ] + l[2] * particles_vel_previous_p[2*i+1];

    k2_x = particles_vel[2*i] + 0.5*dt*k1_x - (l[0] * u + l[1]* u_p + l[2] * u_pp);
    k2_y = particles_vel[2*i+1] + 0.5*dt*k1_y - (l[0] * v + l[1] * v_p + l[2] * v_pp);


    // Compute k3

    device_hermite_interpolate_dx_dy(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
    u = psi_y;
    v = -psi_x;

    k3_x = particles_vel[2*i]  - dt*k1_x + 2*dt*k2_x - u;
    k3_y = particles_vel[2*i + 1] - dt*k1_y + 2*dt*k2_y - v;


    particles_vel[2 * i] =  particles_vel[2 * i] - (dt/tau_p) * (k1_x + 4*k2_x + k3_x)/6;

    particles_pos[2 * i] = fmod(particles_pos[2 * i] + dt * (particles_vel[2*i]), twoPI);
    particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);

    //particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - dt * (particles_vel[2 * i + 1] + psi_x_p) / tau_p;
    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - (dt/tau_p) * (k1_y + 4*k2_y + k3_y)/6;

    particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1]  + dt * (particles_vel[2 * i + 1]), twoPI);
    particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);

  }
}














/*
__ 0.5*dt*global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi, int N, int NX, int NY){
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





