#include "cudaparticle2d.h"
#include "../hermite/cudahermite2d.h"
#include "../grid/cudagrid2d.h"



/*******************************************************************
*						  Particule advection        			   *
*******************************************************************/


__global__ void Rescale(int Nb_particle, double s, double *particles_pos){  // Rescaling the particles uniform distribution on the square [0,s]x[0,s] (s = twoPI)

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

    particles_pos[2 * i] = s * particles_pos[2 * i];
    particles_pos[2 * i + 1] = s * particles_pos[2 * i + 1];
}



// Advect particles using Hermite interpolation and RK2 for the time scheme.

__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi, double *psi_previous, double *psi_previous_p, int N, int NX, int NY, double h, int particles_time_integration_num){
  
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = NX*h;
	double LY = NY*h;

	double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
	double k1_x, k1_y, k2_x, k2_y;

	// compute new position, important: psi is set at t_n+1, psi_previous at t_n and pri_previous_p at t_n-1
	switch (particles_time_integration_num) {
		// euler exp
		case 0: {
			device_hermite_interpolate_dx_dy_1(psi_previous, particles_pos[2*i], particles_pos[2*i+1], &u, &v, NX, NY, h);
			// x_n+1 = x_n + dt*u_n
			particles_pos[2*i  ] += dt * u;
			particles_pos[2*i+1] += dt * v;
			break;
		}
		// heun / modified euler / trapezoidal method
		case 1: {
			// k1 = u_tilde(x, t_n ), no interpolation needed
			device_hermite_interpolate_dx_dy_1(psi_previous, particles_pos[2*i], particles_pos[2*i+1], &u_p, &v_p, NX, NY, h);

			// k2 = u_tilde(x + dt k1, t_n + dt)
			device_hermite_interpolate_dx_dy_1(psi, particles_pos[2*i] + dt*u_p, particles_pos[2*i+1] + dt*v_p, &u, &v, NX, NY, h);
			// x_n+q = x_n + dt/2 (k1 + k2)
			particles_pos[2*i  ] += dt * (u + u_p)/2;
			particles_pos[2*i+1] += dt * (v + v_p)/2;
			break;
		}
		// runge kutta 3, for fluid particles the nicolas version uses this one too
		case 2: {
			// k1 = u_tilde(x, t_n ) = u_n
			device_hermite_interpolate_dx_dy_1(psi_previous, particles_pos[2*i], particles_pos[2*i+1], &u_p, &v_p, NX, NY, h);
			k1_x = u_p;
			k1_y = v_p;

			// k2 = u_tilde(x + dt*k1/2, t_n + dt/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1;
			device_hermite_interpolate_dx_dy_3(psi, psi_previous, psi_previous_p, particles_pos[2*i] + dt*k1_x/2.0, particles_pos[2*i+1] + dt*k1_y/2.0, &u, &v, &u_p, &v_p, &u_pp, &v_pp, NX, NY, h);
			// x_n+q = x_n + dt k2
			k2_x = 0.375 * u + 0.75 * u_p + -0.125 * u_pp;
			k2_y = 0.375 * v + 0.75 * v_p + -0.125 * v_pp;

			// k3 = u_tilde(x + k1 dt - 2 k2 dt, t_n+1) = u_n+1
			device_hermite_interpolate_dx_dy_1(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y, &u, &v, NX, NY, h);

			particles_pos[2*i  ] += dt * (k1_x + 4*k2_x + u)/6;
			particles_pos[2*i+1] += dt * (k1_y + 4*k2_y + v)/6;


			break;
		}
		// explicit midpoint rule from nicolas, first order despite two step method
		case -2: {
			double psi_x, psi_y, psi_x_p, psi_y_p;

			device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
			device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

			particles_pos[2 * i] = particles_pos[2 * i] +  0.5 * dt * (psi_y + psi_y_p);
			particles_pos[2 * i + 1] = particles_pos[2 * i + 1] -  0.5 * dt * (psi_x + psi_x_p);
			break;
		}
		// rkthree from nicolas - i don't know if thats correct, haven't checked
		case -3: {
			// need some more variables
			double psi_x, psi_y, psi_x_p, psi_y_p, psi_x_pp, psi_y_pp, k3_x, k3_y;
			double l[3] = {0.375,0.75,-0.125};

	        // Compute k1
	        device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
	        u_p = psi_y_p; v_p = -psi_x_p;
	        k1_x = u_p; k1_y = v_p;

	        // Compute k2
	        device_hermite_interpolate_dx_dy(psi_previous_p, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
	        u_pp = psi_y_pp; v_pp = -psi_x_pp;

	        device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
	        u_p = psi_y_p; v_p = -psi_x_p;

	        device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
	        u = psi_y; v = -psi_x;
	        k2_x = l[0] * u + l[1]* u_p + l[2] * u_pp;
	        k2_y = l[0] * v + l[1] * v_p + l[2] * v_pp;

	        // Compute k3
	        device_hermite_interpolate_dx_dy(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
	        u = psi_y; v = -psi_x;
	        k3_x = u; k3_y = v;

	        particles_pos[2 * i] = particles_pos[2 * i] +  dt * (k1_x + 4*k2_x + k3_x)/6;
	        particles_pos[2 * i + 1] = particles_pos[2 * i + 1] + dt * (k1_y + 4*k2_y + k3_y)/6;
	        break;
		}
		// default: expl euler
		default: {
			device_hermite_interpolate_dx_dy_1(psi_previous, particles_pos[2*i], particles_pos[2*i+1], &u, &v, NX, NY, h);
			particles_pos[2*i] += dt * u; particles_pos[2*i+1] += dt * v; break;
		}
	}
	// map particle position back into domain, inspired by hermite version
	particles_pos[2*i]   -= floor(particles_pos[2*i]/LX)*LX;
	particles_pos[2*i+1] -= floor(particles_pos[2*i+1]/LY)*LX;

	// debugging of particle position
//		printf("Particle number : %d - Position X : %f - Position Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1]);

}


// initialize inertial particles with fluid velocity
__global__ void Particle_advect_iner_ini(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double u, v;  // velocity placeholders
	// compute velocity
	device_hermite_interpolate_dx_dy_1(psi, particles_pos[2*i], particles_pos[2*i+1], &u, &v, NX, NY, h);
	// transcribe velocity
    particles_vel[2*i  ] =  u;
    particles_vel[2*i+1] =  v;
}


__global__ void Particle_advect_inertia_2(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, double *psi_p, double *psi_p_p, int N, int NX, int NY, double h, double tau_p, int particles_time_integration_num) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = NX*h;
	double LY = NY*h;

	double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
	double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;  // step functions for vel
	double kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y;  // step functions for pos

	// compute new velocity, important: psi is set at t_n+1, psi_previous at t_n and pri_previous_p at t_n-1
	switch (particles_time_integration_num) {
		// euler exp
		case 0: {
			// compute u(p_n,t_n)
			device_hermite_interpolate_dx_dy_1(psi_p, particles_pos[2*i], particles_pos[2*i+1], &u, &v, NX, NY, h);

			// p_n+1 = p_n + dt v_n
			particles_pos[2*i  ] += dt * particles_vel[2*i  ];
			particles_pos[2*i+1] += dt * particles_vel[2*i+1];

			// k1 = F(v_n, t_n) = - (v_n - u_n(p_n)) / tau_p
			// x_n+1 = x_n + dt*k1
			particles_vel[2*i  ] -= dt * (particles_vel[2*i  ] - u)/tau_p;
			particles_vel[2*i+1] -= dt * (particles_vel[2*i+1] - v)/tau_p;
			break;
		}
		// heun / modified euler / trapezoidal method
		case 1: {
			// kp1 = F_p(p_n, v_n, t_n) = v_n
			kp1_x = particles_vel[2*i  ];
			kp1_y = particles_vel[2*i+1];
			// compute u(p_n,t_n)
			device_hermite_interpolate_dx_dy_1(psi_p, particles_pos[2*i], particles_pos[2*i+1], &u_p, &v_p, NX, NY, h);
			// kv1 = F_v(p_n, v_n, t_n) = - (v_n - u(p_n, t_n)) / tau_p
			k1_x = - (kp1_x - u_p)/tau_p;
			k1_y = - (kp1_y - v_p)/tau_p;

			// kp2 = F_p(p_n + h kp_1, v_n + h kv_1, t_n+1) = v_n + h kv_1
			kp2_x = kp1_x + dt * k1_x;
			kp2_y = kp1_y + dt * k1_y;

			// compute u(p_n + hkp1, t_n+1)
			device_hermite_interpolate_dx_dy_1(psi, particles_pos[2*i  ] + dt * kp1_x, particles_pos[2*i+1] + dt * kp1_y, &u, &v, NX, NY, h);
			// kv2 = F_v(p_n + h kp_1, v_n + h kv_1, t_n+1) = - (v_n + h kv_1 - u(p_n + hkp1, t_n+1)) / tau_p
			k2_x = - (kp1_x + dt*k1_x - u) / tau_p;
			k2_y = - (kp1_y + dt*k1_y - v) / tau_p;

			// x_n+1 = x_n + dt/2 (k1 + k2)
			particles_pos[2*i  ] += dt/2.0 * (kp1_x + kp2_x);
			particles_pos[2*i+1] += dt/2.0 * (kp2_y + kp2_y);

			// x_n+1 = x_n + dt/2 (k1 + k2)
			particles_vel[2*i  ] += dt/2.0 * (k1_x + k2_x);
			particles_vel[2*i+1] += dt/2.0 * (k1_x + k2_x);
			break;
		}
		// runge kutta 3
		case 2: {
			// kp1 = F_p(p_n, v_n, t_n) = v_n
			kp1_x = particles_vel[2*i  ];
			kp1_y = particles_vel[2*i+1];
			// compute u(p_n,t_n)
			device_hermite_interpolate_dx_dy_1(psi_p, particles_pos[2*i], particles_pos[2*i+1], &u_p, &v_p, NX, NY, h);
			// kv1 = F_v(p_n, v_n, t_n) = - (v_n - u(p_n, t_n)) / tau_p
			k1_x = - (kp1_x - u_p)/tau_p;
			k1_y = - (kp1_y - v_p)/tau_p;

			// kp2 = F_p(p_n + h/2 kp_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			kp2_x = kp1_x + dt/2.0 * k1_x;
			kp2_y = kp1_y + dt/2.0 * k1_y;

			// compute u(p_n + h/2 kp1, t_n+1/2)
			device_hermite_interpolate_dx_dy_3(psi_p_p, psi_p, psi, particles_pos[2*i  ] + dt/2.0 * kp1_x, particles_pos[2*i+1] + dt/2.0 * kp1_y, &u_pp, &v_pp, &u_p, &v_p, &u, &v, NX, NY, h);
			// kv2 = F_v(p_n + h/2 kp_1, v_n + h/2 kv_1, t_n+1/2) = - (v_n + h/2 kv_1 - u(p_n + h/2kp1, t_n+1/2)) / tau_p with u_n+1/2 = 3/8*u_n+1 + 3/4*u_n - 1/8*u_n-1
			k2_x = - (kp1_x + dt/2.0*k1_x - (0.375*u + 0.75*u_p - 0.125*u_pp)) / tau_p;
			k2_y = - (kp1_y + dt/2.0*k1_y - (0.375*v + 0.75*v_p - 0.125*v_pp)) / tau_p;

			// kp3 = F_p(p_n - h kp_1 + 2h kp_2, v_n - h kv_1 + 2h kv_2, t_n+1) = v_n -h* kv_1 + 2h* kv_2
			kp3_x = kp1_x + dt * (-k1_x + 2 * k2_x);
			kp3_y = kp1_y + dt * (-k1_y + 2 * k2_x);

			// compute u(p_n - h kp1 + 2h kp2, t_n+1)
			device_hermite_interpolate_dx_dy_1(psi, particles_pos[2*i  ] + dt*(-kp1_x+2*kp2_x), particles_pos[2*i+1] + dt*(-kp1_y+2*kp2_y), &u, &v, NX, NY, h);
			// kv3 = F_v(p_n - h kp_1 + 2h kp_2, v_n - h kv_1 + 2h kv_2, t_n+1) = - (v_n + h(-kv_1+2kv_2) - u(p_n+1 + h(-kp_1+2kp_2), t_n+1)) / tau_p
			k3_x = - (kp1_x + dt*(-k1_x+2*k2_x) - u) / tau_p;
			k3_y = - (kp1_y + dt*(-k1_y+2*k2_y) - v) / tau_p;

			// x_n+1 = x_n + dt/6 (k1 + 4*k2 + k3)
			particles_pos[2*i  ] += dt/6.0 * (kp1_x + 4*kp2_x + kp3_x);
			particles_pos[2*i+1] += dt/6.0 * (kp2_y + 4*kp2_y + kp3_y);

			// x_n+1 = x_n + dt/6 (k1 + 4*k2 + k3)
			particles_vel[2*i  ] += dt/6.0 * (k1_x + k2_x + k3_x);
			particles_vel[2*i+1] += dt/6.0 * (k1_x + k2_x + k3_y);

			break;
		}
		// explicit midpoint rule from nicolas
		case -2: {
			double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.

		    device_hermite_interpolate_dx_dy(psi_p, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);

		    particles_vel[2 * i] =  particles_vel[2 * i] - 0.5*(dt/tau_p) * ((particles_vel[2 * i] - psi_y_p) + (particles_vel[2*i] + 0.5*dt*(particles_vel[2 * i] - psi_y_p)  - psi_y_p));
		    particles_pos[2 * i] = particles_pos[2 * i] + dt * particles_vel[2 * i];

		    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - 0.5*(dt/tau_p) * (particles_vel[2 * i + 1] + psi_x_p + (particles_vel[2*i + 1 ] + 0.5*dt*(particles_vel[2 * i] + psi_x_p) + psi_x_p));
		    particles_pos[2 * i + 1] = particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1];
			break;
		}
		// runge kutta 3 from nicolas
		case -3: {
			// need some more variables
			double l[3] = {0.375,0.75,-0.125};
			double psi_x, psi_y, psi_x_p, psi_y_p, psi_x_pp, psi_y_pp; // Previous values of the x and y derivative of Psi.

		    // Compute k1
		    device_hermite_interpolate_dx_dy(psi_p, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
		    u_p = psi_y_p; v_p = -psi_x_p;
		    k1_x = particles_vel[2*i] - u_p;
		    k1_y = particles_vel[2*i+1] - v_p;

		    // Compute k2
		    device_hermite_interpolate_dx_dy(psi_p_p, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
		    u_pp = psi_y_pp; v_pp = -psi_x_pp;

		    device_hermite_interpolate_dx_dy(psi_p, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
		    u_p = psi_y_p; v_p = -psi_x_p;

		    device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
		    u = psi_y; v = -psi_x;
		    k2_x = particles_vel[2*i] + 0.5*dt*k1_x - (l[0] * u + l[1]* u_p + l[2] * u_pp);
		    k2_y = particles_vel[2*i+1] + 0.5*dt*k1_y - (l[0] * v + l[1] * v_p + l[2] * v_pp);

		    // Compute k3

		    device_hermite_interpolate_dx_dy(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
		    u = psi_y; v = -psi_x;
		    k3_x = particles_vel[2*i]  - dt*k1_x + 2*dt*k2_x - u;
		    k3_y = particles_vel[2*i + 1] - dt*k1_y + 2*dt*k2_y - v;

		    particles_vel[2 * i] =  particles_vel[2 * i] - (dt/tau_p) * (k1_x + 4*k2_x + k3_x)/6;
		    particles_pos[2 * i] = particles_pos[2 * i] + dt * particles_vel[2*i];

		    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - (dt/tau_p) * (k1_y + 4*k2_y + k3_y)/6;
		    particles_pos[2 * i + 1] = particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1];
			break;
		}
		// default: expl euler
		default: {
			particles_pos[2*i  ] += dt * particles_vel[2*i  ]; particles_pos[2*i+1] += dt * particles_vel[2*i+1];
			device_hermite_interpolate_dx_dy_1(psi_p, particles_pos[2*i], particles_pos[2*i+1], &u, &v, NX, NY, h);
			particles_vel[2*i  ] -= dt * (particles_vel[2*i  ] - u)/tau_p; particles_vel[2*i+1] -= dt * (particles_vel[2*i+1] - v)/tau_p;
			break;
		}
	}
	// map particle position back into domain, inspired by hermite version
	particles_pos[2*i]   -= floor(particles_pos[2*i]/LX)*LX;
	particles_pos[2*i+1] -= floor(particles_pos[2*i+1]/LY)*LX;

//	// debugging of particle position
//	printf("Particle number : %d - Position X : %f - Position Y : %f - Vel X : %f - Vel Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1], particles_vel[2*i], particles_vel[2*i+1]);

}


//__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, double *psi_previous, int N, int NX, int NY, double h, double tau_p){
//
//  int i;
//  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
//  int Di = blockDim.x * gridDim.x;
//  //double psi_x, psi_y;
//  double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.
//  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
//
//  for(i = pos;i<pos + step_pos;i++){
//
//    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
//    //device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);
//
//    //particles_vel[2 * i] =  particles_vel[2 * i] - dt * (particles_vel[2 * i] - psi_y_p) / tau_p;
//    particles_vel[2 * i] =  particles_vel[2 * i] - 0.5*(dt/tau_p) * ((particles_vel[2 * i] - psi_y_p) + (particles_vel[2*i] + 0.5*dt*(particles_vel[2 * i] - psi_y_p)  - psi_y_p));
//
//    particles_pos[2 * i] = fmod(particles_pos[2 * i] + dt * particles_vel[2 * i], twoPI);
//    particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);
//
//    //particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - dt * (particles_vel[2 * i + 1] + psi_x_p) / tau_p;
//    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - 0.5*(dt/tau_p) * (particles_vel[2 * i + 1] + psi_x_p + (particles_vel[2*i + 1 ] + 0.5*dt*(particles_vel[2 * i] + psi_x_p) + psi_x_p));
//
//    particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1], twoPI);
//    particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);
//
//
//  }
//}
//
//
//__global__ void Particle_advect_inertia_RK3(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, double *psi_previous, double *psi_previous_p, int N, int NX, int NY, double h, double tau_p){
//
//  int i;
//  int thread_finder  = threadIdx.x + blockDim.x * blockIdx.x;
//  int Di = blockDim.x * gridDim.x;
//  //double psi_x, psi_y;
//  double psi_x, psi_y, psi_x_p, psi_y_p, psi_x_pp, psi_y_pp; // Previous values of the x and y derivative of Psi.
//  double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
//  double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;
//  //double vix, viy;
//  double l[3] = {0.375,0.75,-0.125};
//
//  int pos = Nb_particle / Di * thread_finder, step_pos = Nb_particle / Di;
//
//  for(i = pos;i<pos + step_pos;i++){
//
//    // Compute k1
//    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
//    u_p = psi_y_p;
//    v_p = -psi_x_p;
//
//    k1_x = particles_vel[2*i] - u_p;
//    k1_y = particles_vel[2*i+1] - v_p;
//
//    // Compute k2
//    device_hermite_interpolate_dx_dy(psi_previous_p, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
//    u_pp = psi_y_pp;
//    v_pp = -psi_x_pp;
//
//    device_hermite_interpolate_dx_dy(psi_previous, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
//    u_p = psi_y_p;
//    v_p = -psi_x_p;
//
//    device_hermite_interpolate_dx_dy(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
//    u = psi_y;
//    v = -psi_x;
//
//    //k2_x = L1(t+dt/2, t, t-dt, t-2*dt) * u + L2(t+dt/2, t, t-dt, t-2*dt)* u_p + L3(t+dt/2, t, t-dt, t-2*dt)*u_p_p;
//    //k2_y = L1(t+dt/2, t, t-dt, t-2*dt) * v + L2(t+dt/2, t, t-dt, t-2*dt)* v_p + L3(t+dt/2, t, t-dt, t-2*dt)*v_p_p;
//
//    //vix = l[0] * particles_vel[2*i] + l[1]* particles_vel_previous[2*i] + l[2] * particles_vel_previous_p[2*i];
//    //viy = l[0] * particles_vel[2*i + 1] + l[1]* particles_vel_previous[2*i + 1 ] + l[2] * particles_vel_previous_p[2*i+1];
//
//    k2_x = particles_vel[2*i] + 0.5*dt*k1_x - (l[0] * u + l[1]* u_p + l[2] * u_pp);
//    k2_y = particles_vel[2*i+1] + 0.5*dt*k1_y - (l[0] * v + l[1] * v_p + l[2] * v_pp);
//
//
//    // Compute k3
//
//    device_hermite_interpolate_dx_dy(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
//    u = psi_y;
//    v = -psi_x;
//
//    k3_x = particles_vel[2*i]  - dt*k1_x + 2*dt*k2_x - u;
//    k3_y = particles_vel[2*i + 1] - dt*k1_y + 2*dt*k2_y - v;
//
//
//    particles_vel[2 * i] =  particles_vel[2 * i] - (dt/tau_p) * (k1_x + 4*k2_x + k3_x)/6;
//
//    particles_pos[2 * i] = fmod(particles_pos[2 * i] + dt * (particles_vel[2*i]), twoPI);
//    particles_pos[2 * i] =  (particles_pos[2 * i] < 0)*(twoPI + particles_pos[2 * i]) + (particles_pos[2 * i] > 0)*(particles_pos[2 * i]);
//
//    //particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - dt * (particles_vel[2 * i + 1] + psi_x_p) / tau_p;
//    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - (dt/tau_p) * (k1_y + 4*k2_y + k3_y)/6;
//
//    particles_pos[2 * i + 1] = fmod(particles_pos[2 * i + 1]  + dt * (particles_vel[2 * i + 1]), twoPI);
//    particles_pos[2 * i + 1] = (particles_pos[2 * i + 1] < 0)*(twoPI + particles_pos[2 * i + 1]) + (particles_pos[2 * i + 1] > 0)*(particles_pos[2 * i + 1]);
//
//  }
//}














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





