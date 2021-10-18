#include "cmm-particles.h"

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

__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi, int N, int NX, int NY, double h, int particles_time_integration_num){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = NX*h;
	double LY = NY*h;

//	double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
//	double k1_x, k1_y, k2_x, k2_y;

	// compute new position, important: psi is set at t_n+1, psi_previous at t_n and pri_previous_p at t_n-1
	switch (particles_time_integration_num) {
		// euler exp
		case 10: {
			double u[2];
			device_hermite_interpolate_grad_2D(psi + 4*N, particles_pos + 2*i, u, NX, NY, h, 1);
			// x_n+1 = x_n + dt*u_n
			particles_pos[2*i  ] += dt * u[0];
			particles_pos[2*i+1] += dt * u[1];
			break;
		}
		// heun / modified euler / trapezoidal method
		case 20: {
			double u[4];
			// k1 = u_tilde(x, t_n ), no interpolation needed
			device_hermite_interpolate_grad_2D(psi + 4*N, particles_pos + 2*i, u+2, NX, NY, h, 1);

			// k2 = u_tilde(x + dt k1, t_n + dt)
			double k2[2] = {particles_pos[2*i] + dt*u[2], particles_pos[2*i+1] + dt*u[3]};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 1);
			// x_n+q = x_n + dt/2 (k1 + k2)
			particles_pos[2*i  ] += dt * (u[0] + u[2])/2;
			particles_pos[2*i+1] += dt * (u[1] + u[3])/2;
			break;
		}
		// runge kutta 3 for fluid particles, the nicolas version is identical
		case 30: {
			double u[6];
			// k1 = u_tilde(x, t_n ) = u_n
			device_hermite_interpolate_grad_2D(psi + 4*N, particles_pos + 2*i, u+2, NX, NY, h, 1);
			double k1[2] = {u[2], u[3]};

			// k2 = u_tilde(x + dt*k1/2, t_n + dt/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1
			double k2[2] = {particles_pos[2*i] + dt*k1[0]/2.0, particles_pos[2*i+1] + dt*k1[1]/2.0};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 3);
			// x_n+q = x_n + dt k2
			k2[0] = 0.375 * u[0] + 0.75 * u[2] + -0.125 * u[4];
			k2[1] = 0.375 * u[1] + 0.75 * u[3] + -0.125 * u[5];

			// k3 = u_tilde(x + k1 dt - 2 k2 dt, t_n+1) = u_n+1
			double k3[2] = {particles_pos[2*i] - dt*k1[0] + 2*dt*k2[0], particles_pos[2*i+1] - dt*k1[1] + 2*dt*k2[1]};
			device_hermite_interpolate_grad_2D(psi, k3, u, NX, NY, h, 1);

			particles_pos[2*i  ] += dt * (k1[0] + 4*k2[0] + u[0])/6;
			particles_pos[2*i+1] += dt * (k1[1] + 4*k2[1] + u[1])/6;


			break;
		}
		// explicit midpoint rule from nicolas, first order despite two step method
		case 25: {
			double psi_x, psi_y, psi_x_p, psi_y_p;

			device_hermite_interpolate_dx_dy_2D(psi + 4*N, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
			device_hermite_interpolate_dx_dy_2D(psi, particles_pos[2 * i] + 0.5 * dt * psi_y_p, particles_pos[2 * i + 1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

			particles_pos[2 * i] = particles_pos[2 * i] +  0.5 * dt * (psi_y + psi_y_p);
			particles_pos[2 * i + 1] = particles_pos[2 * i + 1] -  0.5 * dt * (psi_x + psi_x_p);
			break;
		}
		// rkthree from nicolas - i don't know if thats correct, haven't checked
		case 35: {
			// need some more variables
			double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
			double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;
			double psi_x, psi_y, psi_x_p, psi_y_p, psi_x_pp, psi_y_pp;
			double l[3] = {0.375,0.75,-0.125};

	        // Compute k1
	        device_hermite_interpolate_dx_dy_2D(psi + 4*N, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
	        u_p = psi_y_p; v_p = -psi_x_p;
	        k1_x = u_p; k1_y = v_p;

	        // Compute k2
	        device_hermite_interpolate_dx_dy_2D(psi + 8*N, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
	        u_pp = psi_y_pp; v_pp = -psi_x_pp;

	        device_hermite_interpolate_dx_dy_2D(psi + 4*N, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
	        u_p = psi_y_p; v_p = -psi_x_p;

	        device_hermite_interpolate_dx_dy_2D(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
	        u = psi_y; v = -psi_x;
	        k2_x = l[0] * u + l[1]* u_p + l[2] * u_pp;
	        k2_y = l[0] * v + l[1] * v_p + l[2] * v_pp;

	        // Compute k3
	        device_hermite_interpolate_dx_dy_2D(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
	        u = psi_y; v = -psi_x;
	        k3_x = u; k3_y = v;

	        particles_pos[2 * i] = particles_pos[2 * i] +  dt * (k1_x + 4*k2_x + k3_x)/6;
	        particles_pos[2 * i + 1] = particles_pos[2 * i + 1] + dt * (k1_y + 4*k2_y + k3_y)/6;
	        break;
		}
		// default: zero
		default: { particles_pos[2*i  ] = particles_pos[2*i+1] = 0; break; }
	}
	// map particle position back into domain, inspired by hermite version
	particles_pos[2*i]   -= floor(particles_pos[2*i]/LX)*LX;
	particles_pos[2*i+1] -= floor(particles_pos[2*i+1]/LY)*LX;

//	// debugging of particle position
//	if (i == 0) printf("Particle number : %d - Position X : %f - Position Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1]);

}


// initialize inertial particles with fluid velocity
__global__ void Particle_advect_inertia_init(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double u[2];  // velocity placeholders
	// compute velocity
	device_hermite_interpolate_grad_2D(psi, particles_pos + 2*i, u, NX, NY, h, 1);
	// transcribe velocity
    particles_vel[2*i  ] =  u[0];
    particles_vel[2*i+1] =  u[1];
}


__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h, double tau_p, int particles_time_integration_num) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = NX*h;
	double LY = NY*h;

//	double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
//	double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;  // step functions for vel
//	double kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y;  // step functions for pos

	// compute new velocity, important: psi is set at t_n+1, psi_previous at t_n and pri_previous_p at t_n-1
	switch (particles_time_integration_num) {
		// euler exp
		case 10: {
			double u[2];
			// compute u(p_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, particles_pos + 2*i, u, NX, NY, h, 1);

			// p_n+1 = p_n + dt v_n
			particles_pos[2*i  ] += dt * particles_vel[2*i  ];
			particles_pos[2*i+1] += dt * particles_vel[2*i+1];

			// k1 = F(v_n, t_n) = - (v_n - u_n(p_n)) / tau_p
			// x_n+1 = x_n + dt*k1
			particles_vel[2*i  ] -= dt * (particles_vel[2*i  ] - u[0])/tau_p;
			particles_vel[2*i+1] -= dt * (particles_vel[2*i+1] - u[1])/tau_p;
			break;
		}
		// heun / modified euler / trapezoidal method
		case 20: {
			double u[4];
			// kp1 = F_p(p_n, v_n, t_n) = v_n
			double kp1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(p_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, particles_pos + 2*i, u+2, NX, NY, h, 1);
			// kv1 = F_v(p_n, v_n, t_n) = - (v_n - u(p_n, t_n)) / tau_p
			double k1[2] = {- (kp1[0] - u[2])/tau_p, - (kp1[1] - u[3])/tau_p};

			// kp2 = F_p(p_n + h kp_1, v_n + h kv_1, t_n+1) = v_n + h kv_1
			double kp2[2] = {kp1[0] + dt * k1[0], kp1[1] + dt * k1[1]};

			// compute u(p_n + hkp1, t_n+1)
			double k2[2] = {particles_pos[2*i  ] + dt * kp1[0], particles_pos[2*i+1] + dt * kp1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 1);
			// kv2 = F_v(p_n + h kp_1, v_n + h kv_1, t_n+1) = - (v_n + h kv_1 - u(p_n + hkp1, t_n+1)) / tau_p
			k2[0] = - (kp1[0] + dt*k1[0] - u[0]) / tau_p;
			k2[1] = - (kp1[1] + dt*k1[1] - u[1]) / tau_p;

			// x_n+1 = x_n + dt/2 (k1 + k2)
			particles_pos[2*i  ] += dt/2.0 * (kp1[0] + kp2[0]);
			particles_pos[2*i+1] += dt/2.0 * (kp2[1] + kp2[1]);

			// x_n+1 = x_n + dt/2 (k1 + k2)
			particles_vel[2*i  ] += dt/2.0 * (k1[0] + k2[0]);
			particles_vel[2*i+1] += dt/2.0 * (k1[1] + k2[1]);
			break;
		}
		// runge kutta 3
		case 30: {
			double u[6];
			// kp1 = F_p(p_n, v_n, t_n) = v_n
			double kp1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(p_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, particles_pos + 2*i, u+2, NX, NY, h, 1);
			// kv1 = F_v(p_n, v_n, t_n) = - (v_n - u(p_n, t_n)) / tau_p
			double k1[2] = {- (kp1[0] - u[2])/tau_p, - (kp1[1] - u[3])/tau_p};

			// kp2 = F_p(p_n + h/2 kp_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			double kp2[2] = {kp1[0] + dt/2.0 * k1[0], kp1[1] + dt/2.0 * k1[1]};

			// compute u(p_n + h/2 kp1, t_n+1/2)
			double k2[2] = {particles_pos[2*i  ] + dt/2.0 * kp1[0], particles_pos[2*i+1] + dt/2.0 * kp1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 3);
			// kv2 = F_v(p_n + h/2 kp_1, v_n + h/2 kv_1, t_n+1/2) = - (v_n + h/2 kv_1 - u(p_n + h/2kp1, t_n+1/2)) / tau_p with u_n+1/2 = 3/8*u_n+1 + 3/4*u_n - 1/8*u_n-1
			k2[0] = - (kp1[0] + dt/2.0*k1[0] - (0.375 * u[0] + 0.75 * u[2] + -0.125 * u[4])) / tau_p;
			k2[1] = - (kp1[1] + dt/2.0*k1[1] - (0.375 * u[1] + 0.75 * u[3] + -0.125 * u[5])) / tau_p;

			// kp3 = F_p(p_n - h kp_1 + 2h kp_2, v_n - h kv_1 + 2h kv_2, t_n+1) = v_n -h* kv_1 + 2h* kv_2
			double kp3[2] = {kp1[0] + dt * (-k1[0] + 2 * k2[0]), kp1[1] + dt * (-k1[1] + 2 * k2[1])};

			// compute u(p_n - h kp1 + 2h kp2, t_n+1)
			double k3[2] = {particles_pos[2*i  ] + dt*(-kp1[0]+2*kp2[0]), particles_pos[2*i+1] + dt*(-kp1[1]+2*kp2[1])};
			device_hermite_interpolate_grad_2D(psi, k3, u, NX, NY, h, 1);
			// kv3 = F_v(p_n - h kp_1 + 2h kp_2, v_n - h kv_1 + 2h kv_2, t_n+1) = - (v_n + h(-kv_1+2kv_2) - u(p_n+1 + h(-kp_1+2kp_2), t_n+1)) / tau_p
			k3[0] = - (kp1[0] + dt*(-k1[0]+2*k2[0]) - u[0]) / tau_p;
			k3[1] = - (kp1[1] + dt*(-k1[1]+2*k2[1]) - u[1]) / tau_p;

			// x_n+1 = x_n + dt/6 (k1 + 4*k2 + k3)
			particles_pos[2*i  ] += dt/6.0 * (kp1[0] + 4*kp2[0] + kp3[0]);
			particles_pos[2*i+1] += dt/6.0 * (kp2[1] + 4*kp2[1] + kp3[1]);

			// x_n+1 = x_n + dt/6 (k1 + 4*k2 + k3)
			particles_vel[2*i  ] += dt/6.0 * (k1[0] + k2[0] + k3[0]);
			particles_vel[2*i+1] += dt/6.0 * (k1[1] + k2[1] + k3[1]);

			break;
		}
		// explicit midpoint rule from nicolas
		case 25: {
			double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.

		    device_hermite_interpolate_dx_dy_2D(psi + 4*N, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);

		    particles_vel[2 * i] =  particles_vel[2 * i] - 0.5*(dt/tau_p) * ((particles_vel[2 * i] - psi_y_p) + (particles_vel[2*i] + 0.5*dt*(particles_vel[2 * i] - psi_y_p)  - psi_y_p));
		    particles_pos[2 * i] = particles_pos[2 * i] + dt * particles_vel[2 * i];

		    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - 0.5*(dt/tau_p) * (particles_vel[2 * i + 1] + psi_x_p + (particles_vel[2*i + 1 ] + 0.5*dt*(particles_vel[2 * i] + psi_x_p) + psi_x_p));
		    particles_pos[2 * i + 1] = particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1];
			break;
		}
		// runge kutta 3 from nicolas
		case 35: {
			// need some more variables
			double u, v, u_p, v_p, u_pp, v_pp; // Previous values of the x and y derivative of Psi.
			double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;  // step functions for vel
			double l[3] = {0.375,0.75,-0.125};
			double psi_x, psi_y, psi_x_p, psi_y_p, psi_x_pp, psi_y_pp; // Previous values of the x and y derivative of Psi.

		    // Compute k1
		    device_hermite_interpolate_dx_dy_2D(psi + 4*N, particles_pos[2 * i], particles_pos[2 * i + 1], &psi_x_p, &psi_y_p, NX, NY, h);
		    u_p = psi_y_p; v_p = -psi_x_p;
		    k1_x = particles_vel[2*i] - u_p;
		    k1_y = particles_vel[2*i+1] - v_p;

		    // Compute k2
		    device_hermite_interpolate_dx_dy_2D(psi + 8*N, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
		    u_pp = psi_y_pp; v_pp = -psi_x_pp;

		    device_hermite_interpolate_dx_dy_2D(psi + 4*N, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
		    u_p = psi_y_p; v_p = -psi_x_p;

		    device_hermite_interpolate_dx_dy_2D(psi, particles_pos[2 * i] + 0.5 * dt * k1_x, particles_pos[2 * i + 1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
		    u = psi_y; v = -psi_x;
		    k2_x = particles_vel[2*i] + 0.5*dt*k1_x - (l[0] * u + l[1]* u_p + l[2] * u_pp);
		    k2_y = particles_vel[2*i+1] + 0.5*dt*k1_y - (l[0] * v + l[1] * v_p + l[2] * v_pp);

		    // Compute k3

		    device_hermite_interpolate_dx_dy_2D(psi, particles_pos[2*i] - dt*k1_x + 2*dt*k2_x, particles_pos[2*i+1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
		    u = psi_y; v = -psi_x;
		    k3_x = particles_vel[2*i]  - dt*k1_x + 2*dt*k2_x - u;
		    k3_y = particles_vel[2*i + 1] - dt*k1_y + 2*dt*k2_y - v;

		    particles_vel[2 * i] =  particles_vel[2 * i] - (dt/tau_p) * (k1_x + 4*k2_x + k3_x)/6;
		    particles_pos[2 * i] = particles_pos[2 * i] + dt * particles_vel[2*i];

		    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - (dt/tau_p) * (k1_y + 4*k2_y + k3_y)/6;
		    particles_pos[2 * i + 1] = particles_pos[2 * i + 1]  + dt * particles_vel[2 * i + 1];
			break;
		}
		// default: zero
		default: { particles_pos[2*i  ] = particles_pos[2*i+1] = 0; particles_vel[2*i  ] = particles_vel[2*i+1] = 0; break; }
	}
	// map particle position back into domain, inspired by hermite version
	particles_pos[2*i]   -= floor(particles_pos[2*i]/LX)*LX;
	particles_pos[2*i+1] -= floor(particles_pos[2*i+1]/LY)*LX;

	// debugging of particle position
//	if (i == 0) printf("Particle number : %d - Position X : %f - Position Y : %f - Vel X : %f - Vel Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1], particles_vel[2*i], particles_vel[2*i+1]);

}
