#include "cmm-particles.h"

//#include "stdio.h"

/*******************************************************************
*						  Particule advection        			   *
*******************************************************************/


__global__ void Rescale(int Nb_particle, double s, double *particles_pos){  // Rescaling the particles uniform distribution on the square [0,s]x[0,s] (s = twoPI)

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

    particles_pos[2 * i] *= s;
    particles_pos[2 * i + 1] *= s;
}



// Advect particles using Hermite interpolation and RK2 for the time scheme.

__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi, int N, int NX, int NY, double h, int particles_time_integration_num){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = NX*h;
	double LY = NY*h;

	// reading globally is expensive, so we only want to do it once
	double part_pos_old[2] = { particles_pos[2*i  ], particles_pos[2*i+1] };

	// compute new position, important: psi is set at t_n+1, psi_previous at t_n and pri_previous_p at t_n-1
	switch (particles_time_integration_num) {
		// euler exp
		case 10: {
			double u[2];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u, NX, NY, h, 1);

			// x_n+1 = x_n + dt*u_n
			part_pos_old[0] += dt * u[0];
			part_pos_old[1] += dt * u[1];
			break;
		}
		// heun / modified euler / trapezoidal method
		case 20: {
			double k[4];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, k, NX, NY, h, 1);

			// k2 = f_x(x_n + h*k1, t_n ) = u_n+1 ( x_n + h*k1 )
			k[2] = part_pos_old[0] + dt*k[0]; k[3] = part_pos_old[1] + dt*k[1];
			device_hermite_interpolate_grad_2D(psi, k+2, k+2, NX, NY, h, 1);

			// x_n+1 = x_n + dt/2 (k1 + k2)
			part_pos_old[0] += dt * (k[0] + k[2])/2;
			part_pos_old[1] += dt * (k[1] + k[3])/2;
			break;
		}
		// runge kutta 3 for fluid particles, the nicolas version is identical
		case 30: {
			double u[6];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u+2, NX, NY, h, 1);
			double k1[2] = {u[2], u[3]};

			// k2 = f_x(x_n + h/2*k1, t_n+1/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1 ( x_n + h/2*k1 )
			double k2[2] = {part_pos_old[0] + dt*k1[0]/2.0, part_pos_old[1] + dt*k1[1]/2.0};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 3);
			k2[0] = 0.375 * u[0] + 0.75 * u[2] + -0.125 * u[4];
			k2[1] = 0.375 * u[1] + 0.75 * u[3] + -0.125 * u[5];

			// k3 = f_x(x_n - h*k1 + 2h*k2, t_n+1) = u_n+1 ( x_n - h*k1 + 2h*k2, t_n+1 )
			double k3[2] = {part_pos_old[0] - dt*k1[0] + 2*dt*k2[0], part_pos_old[1] - dt*k1[1] + 2*dt*k2[1]};
			device_hermite_interpolate_grad_2D(psi, k3, u, NX, NY, h, 1);

			// x_n+q = x_n + dt/6 (k1 + 4*k2 + k3)
			part_pos_old[0] += dt * (k1[0] + 4*k2[0] + u[0])/6;
			part_pos_old[1] += dt * (k1[1] + 4*k2[1] + u[1])/6;

			break;
		}
		// runge kutta 4 for fluid particles
		case 40: {
			double u[8];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u+2, NX, NY, h, 1);
			double k1[2] = {u[2], u[3]};

			// k2 = f_x(x_n + h/2*k1, t_n+1/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1 ( x_n + h/2*k1 )
			double k2[2] = {part_pos_old[0] + dt*k1[0]/2.0, part_pos_old[1] + dt*k1[1]/2.0};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 4);
			k2[0] = 0.3125 * u[0] + 0.9375 * u[2] + -0.3125 * u[4] + 0.0625 * u[6];
			k2[1] = 0.3125 * u[1] + 0.9375 * u[3] + -0.3125 * u[5] + 0.0625 * u[7];

			// k2 = f_x(x_n + h/2*k2, t_n+1/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1 ( x_n + h/2*k2 )
			double k3[2] = {part_pos_old[0] + dt*k2[0]/2.0, part_pos_old[1] + dt*k2[1]/2.0};
			device_hermite_interpolate_grad_2D(psi, k3, u, NX, NY, h, 4);
			k3[0] = 0.3125 * u[0] + 0.9375 * u[2] + -0.3125 * u[4] + 0.0625 * u[6];
			k3[1] = 0.3125 * u[1] + 0.9375 * u[3] + -0.3125 * u[5] + 0.0625 * u[7];

			// k4 = f_x(x_n + h*k3, t_n ) = u_n+1 ( x_n + h*k3 )
			double k4[2] = {part_pos_old[0] + dt*k3[0], part_pos_old[1] + dt*k3[1]};
			device_hermite_interpolate_grad_2D(psi, k4, u, NX, NY, h, 1);

			// x_n+1 = x_n + dt/6 (k1 + 2*k2 + 2*k3 + k4)
			part_pos_old[0] += dt * (k1[0] + 2*k2[0] + 2*k3[0] + u[0])/6;
			part_pos_old[1] += dt * (k1[1] + 2*k2[1] + 2*k3[1] + u[1])/6;

			break;
		}
		// modified rkthree with optimal timesteps chosen to result in no lagrange interpolation
		case 31: {
			double k[6];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, k, NX, NY, h, 1);

			// k2 = f_x(x_n + h*k1, t_n ) = u_n+1 ( x_n + h*k1 )
			k[2] = part_pos_old[0] + dt*k[0]; k[3] = part_pos_old[1] + dt*k[1];
			device_hermite_interpolate_grad_2D(psi, k+2, k+2, NX, NY, h, 1);

			// k3 = f_x(x_n + h*k1 - 2h*k2, t_n-1) = u_n-1 ( x_n + h*k1 - 2h*k2 )
			k[4] = part_pos_old[0] + dt*(k[0] - 2*k[2]); k[5] = part_pos_old[1] + dt*(k[1] - 2*k[3]);
			device_hermite_interpolate_grad_2D(psi + 8*N, k+4, k+4, NX, NY, h, 1);

			// x_n+1 = x_n + dt/12 (8*k1 + 5*k2 - k3)
			part_pos_old[0] += dt * (8*k[0] + 5*k[2] - k[4])/12.0;
			part_pos_old[1] += dt * (8*k[1] + 5*k[3] - k[5])/12.0;
			break;
		}
		// modified runge kutta 4 for fluid particles with optimal timesteps chosen to result in only ine lagrange interpolation
		case 41: {
			double u[8], k[8];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, k, NX, NY, h, 1);

			// k2 = f_x(x_n + h*k1, t_n ) = u_n+1 ( x_n + h*k1 )
			k[2] = part_pos_old[0] + dt*k[0]; k[3] = part_pos_old[1] + dt*k[1];
			device_hermite_interpolate_grad_2D(psi, k+2, k+2, NX, NY, h, 1);

			// k3 = f_x(x_n + 3h/8*k1 + h/8*k2, t_n+1/2) = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2 ( x_n + 3h/8*k1 + h/8*k2 )
			k[4] = part_pos_old[0] + dt/8.0*(3*k[0] + k[2]); k[5] = part_pos_old[1] + dt/8.0*(3*k[1] + k[3]);
			device_hermite_interpolate_grad_2D(psi, k+4, u, NX, NY, h, 4);
			k[4] = 0.3125 * u[0] + 0.9375 * u[2] + -0.3125 * u[4] + 0.0625 * u[6];
			k[5] = 0.3125 * u[1] + 0.9375 * u[3] + -0.3125 * u[5] + 0.0625 * u[7];

			// k4 = f_x(x_n - h/3*k2 + 4h/3 k3, t_n+1) = u_n+1 ( x_n - h/3*k2 + 4h/3 k3 )
			k[6] = part_pos_old[0] + dt/3.0*(-k[2] + 4*k[4]); k[7] = part_pos_old[1] + dt/3.0*(-k[3] + 4*k[5]);
			device_hermite_interpolate_grad_2D(psi, k+6, k+6, NX, NY, h, 1);

			// x_n+1 = x_n + dt/12 (2*k1 - k2 + 8*k3 + 3*k4)
			part_pos_old[0] += dt * (2*k[0] - k[2] + 8*k[4] + 3*k[6])/12.0;
			part_pos_old[1] += dt * (2*k[1] - k[3] + 8*k[5] + 3*k[7])/12.0;

			break;
		}
		// explicit midpoint rule from nicolas, first order despite two step method
		case 25: {
			double psi_x, psi_y, psi_x_p, psi_y_p;

			device_hermite_interpolate_dx_dy_2D(psi + 4*N, part_pos_old[0], part_pos_old[1], &psi_x_p, &psi_y_p, NX, NY, h);
			device_hermite_interpolate_dx_dy_2D(psi, part_pos_old[0] + 0.5 * dt * psi_y_p, part_pos_old[1] - 0.5 * dt * psi_x_p, &psi_x, &psi_y, NX, NY, h);

			part_pos_old[0] +=  0.5 * dt * (psi_y + psi_y_p);
			part_pos_old[1] +=  0.5 * dt * (psi_x + psi_x_p);
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
	        device_hermite_interpolate_dx_dy_2D(psi + 4*N, part_pos_old[0], part_pos_old[1], &psi_x_p, &psi_y_p, NX, NY, h);
	        u_p = psi_y_p; v_p = -psi_x_p;
	        k1_x = u_p; k1_y = v_p;

	        // Compute k2
	        device_hermite_interpolate_dx_dy_2D(psi + 8*N, part_pos_old[0] + 0.5 * dt * k1_x, part_pos_old[1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
	        u_pp = psi_y_pp; v_pp = -psi_x_pp;

	        device_hermite_interpolate_dx_dy_2D(psi + 4*N, part_pos_old[0] + 0.5 * dt * k1_x, part_pos_old[1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
	        u_p = psi_y_p; v_p = -psi_x_p;

	        device_hermite_interpolate_dx_dy_2D(psi, part_pos_old[0] + 0.5 * dt * k1_x, part_pos_old[1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
	        u = psi_y; v = -psi_x;
	        k2_x = l[0] * u + l[1]* u_p + l[2] * u_pp;
	        k2_y = l[0] * v + l[1] * v_p + l[2] * v_pp;

	        // Compute k3
	        device_hermite_interpolate_dx_dy_2D(psi, part_pos_old[0] - dt*k1_x + 2*dt*k2_x, part_pos_old[1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
	        u = psi_y; v = -psi_x;
	        k3_x = u; k3_y = v;

	        part_pos_old[0] +=  dt * (k1_x + 4*k2_x + k3_x)/6;
	        part_pos_old[1] += dt * (k1_y + 4*k2_y + k3_y)/6;
	        break;
		}
		// default: zero
		default: { part_pos_old[0] = part_pos_old[1] = 0; break; }
	}
	// map particle position back into domain, inspired by hermite version
	particles_pos[2*i]   = part_pos_old[0] - floor(part_pos_old[0]/LX)*LX;
	particles_pos[2*i+1] = part_pos_old[1] - floor(part_pos_old[1]/LY)*LX;

	// debugging of particle position
//	if (i == 0) printf("Particle number : %d - Position X : %f - Position Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1]);
}


// initialize inertial particles with fluid velocity
__global__ void Particle_advect_inertia_init(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	// compute velocity and transcribe directly
	device_hermite_interpolate_grad_2D(psi, particles_pos + 2*i, particles_pos + 2*i, NX, NY, h, 1);
}


__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, int N, int NX, int NY, double h, double tau_p, int particles_time_integration_num) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = NX*h;
	double LY = NY*h;

	// reading globally is expensive, so we only want to do it once
	double part_pos_old[2] = { particles_pos[2*i  ], particles_pos[2*i+1] };

	// compute new velocity, important: psi is set at t_n+1, psi_previous at t_n ...
	switch (particles_time_integration_num) {
		// euler exp
		case 10: {
			double u[2];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = { particles_vel[2*i  ], particles_vel[2*i+1] };
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u, NX, NY, h, 1);

			// x_n+1 = x_n + dt v_n
			part_pos_old[0] += dt * kx_1[0];
			part_pos_old[1] += dt * kx_1[1];

			// kv1 = F_v(v_n, t_n) = - (v_n - u_n(x_n)) / tau_p
			// v_n+1 = v_n + dt*kv1
			particles_vel[2*i  ] -= dt * (kx_1[0] - u[0])/tau_p;
			particles_vel[2*i+1] -= dt * (kx_1[1] - u[1])/tau_p;
			break;
		}
		// heun / modified euler / trapezoidal method
		case 20: {
			double u[4];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u+2, NX, NY, h, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			double k1[2] = {- (kx_1[0] - u[2])/tau_p, - (kx_1[1] - u[3])/tau_p};

			// kx_2 = F_x(x_n + h kx_1, v_n + h kv_1, t_n+1) = v_n + h kv_1
			double kx_2[2] = {kx_1[0] + dt * k1[0], kx_1[1] + dt * k1[1]};

			// compute u(x_n + h kx_1, t_n+1)
			double k2[2] = {part_pos_old[0] + dt * kx_1[0], part_pos_old[1] + dt * kx_1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 1);
			// kv2 = F_v(x_n + h kx_1, v_n + h kv_1, t_n+1) = - (kx_2 - u(x_n + h kx_1, t_n+1)) / tau_p
			k2[0] = - (kx_2[0] - u[0]) / tau_p;
			k2[1] = - (kx_2[1] - u[1]) / tau_p;

			// x_n+1 = x_n + dt/2 (kx_1 + kx_2)
			part_pos_old[0] += dt/2.0 * (kx_1[0] + kx_2[0]);
			part_pos_old[1] += dt/2.0 * (kx_1[1] + kx_2[1]);

			// v_n+1 = v_n + dt/2 (k1 + k2)
			particles_vel[2*i  ] += dt/2.0 * (k1[0] + k2[0]);
			particles_vel[2*i+1] += dt/2.0 * (k1[1] + k2[1]);
			break;
		}
		// classical runge kutta 3
		case 30: {
			double u[6];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u+2, NX, NY, h, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			double k1[2] = {- (kx_1[0] - u[2])/tau_p, - (kx_1[1] - u[3])/tau_p};

			// kx_2 = F_x(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			double kx_2[2] = {kx_1[0] + dt/2.0 * k1[0], kx_1[1] + dt/2.0 * k1[1]};

			// compute u(x_n + h/2 kx_1, t_n+1/2) for all lagrange timesteps
			double k2[2] = {part_pos_old[0] + dt/2.0 * kx_1[0], part_pos_old[1] + dt/2.0 * kx_1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 3);
			// kv2 = F_v(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = - (kx_2 - u(x_n + h/2kx_1, t_n+1/2)) / tau_p with u_n+1/2 = 3/8*u_n+1 + 3/4*u_n - 1/8*u_n-1
			k2[0] = - (kx_2[0] - (0.375 * u[0] + 0.75 * u[2] + -0.125 * u[4])) / tau_p;
			k2[1] = - (kx_2[1] - (0.375 * u[1] + 0.75 * u[3] + -0.125 * u[5])) / tau_p;

			// kx_3 = F_x(x_n - h kx_1 + 2h kx_2, v_n - h kv_1 + 2h kv_2, t_n+1) = v_n -h* kv_1 + 2h* kv_2
			double kx_3[2] = {kx_1[0] + dt * (-k1[0] + 2*k2[0]), kx_1[1] + dt * (-k1[1] + 2*k2[1])};

			// compute u(x_n - h kx_1 + 2h kx_2, t_n+1)
			double k3[2] = {part_pos_old[0] + dt*(-kx_1[0]+2*kx_2[0]), part_pos_old[1] + dt*(-kx_1[1]+2*kx_2[1])};
			device_hermite_interpolate_grad_2D(psi, k3, u, NX, NY, h, 1);
			// kv3 = F_v(x_n - h kx_1 + 2h kx_2, v_n - h kv_1 + 2h kv_2, t_n+1) = - (kx_3 - u(x_n+1 + h(-kx_1+2kx_2), t_n+1)) / tau_p
			k3[0] = - (kx_3[0] - u[0]) / tau_p;
			k3[1] = - (kx_3[1] - u[1]) / tau_p;

			// x_n+1 = x_n + dt/6 (kx_1 + 4*kx_2 + kx_3)
			part_pos_old[0] += dt/6.0 * (kx_1[0] + 4*kx_2[0] + kx_3[0]);
			part_pos_old[1] += dt/6.0 * (kx_1[1] + 4*kx_2[1] + kx_3[1]);

			// v_n+1 = v_n + dt/6 (k1 + 4*k2 + k3)
			particles_vel[2*i  ] += dt/6.0 * (k1[0] + 4*k2[0] + k3[0]);
			particles_vel[2*i+1] += dt/6.0 * (k1[1] + 4*k2[1] + k3[1]);

			break;
		}
		// classical runge kutta 4
		case 40: {
			double u[8];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, u+2, NX, NY, h, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			double k1[2] = {- (kx_1[0] - u[2])/tau_p, - (kx_1[1] - u[3])/tau_p};


			// kx_2 = F_x(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			double kx_2[2] = {kx_1[0] + dt/2.0 * k1[0], kx_1[1] + dt/2.0 * k1[1]};

			// compute u(x_n + h/2 kx_1, t_n+1/2) for all lagrange timesteps
			double k2[2] = {part_pos_old[0] + dt/2.0 * kx_1[0], part_pos_old[1] + dt/2.0 * kx_1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, NX, NY, h, 4);
			// kv2 = F_v(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = - (kx_2 - u(x_n + h/2kx_1, t_n+1/2)) / tau_p with u_n+1/2 = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2
			k2[0] = - (kx_2[0] - (0.3125 * u[0] + 0.9375 * u[2] + -0.3125 * u[4] + 0.0625 * u[6])) / tau_p;
			k2[1] = - (kx_2[1] - (0.3125 * u[1] + 0.9375 * u[3] + -0.3125 * u[5] + 0.0625 * u[7])) / tau_p;


			// kx_3 = F_x(x_n + h/2 kx_2, v_n + h/2 kv_2, t_n+1/2) = v_n + h/2 kv_2
			double kx_3[2] = {kx_1[0] + dt/2.0 * k2[0], kx_1[1] + dt/2.0 * k2[1]};

			// compute u(x_n + h/2 kx_2, t_n+1/2) for all lagrange timesteps
			double k3[2] = {part_pos_old[0] + dt/2.0 * kx_2[0], part_pos_old[1] + dt/2.0 * kx_2[1]};
			device_hermite_interpolate_grad_2D(psi, k3, u, NX, NY, h, 4);
			// kv3 = F_v(x_n + h/2 kx_2, v_n + h/2 kv_2, t_n+1/2) = - (kx_3 - u(x_n + h/2kx_2, t_n+1/2)) / tau_p with u_n+1/2 = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2
			k3[0] = - (kx_3[0] - (0.3125 * u[0] + 0.9375 * u[2] + -0.3125 * u[4] + 0.0625 * u[6])) / tau_p;
			k3[1] = - (kx_3[1] - (0.3125 * u[1] + 0.9375 * u[3] + -0.3125 * u[5] + 0.0625 * u[7])) / tau_p;


			// kx_4 = F_x(x_n + h kx_3, v_n + h kv_3, t_n+1) = v_n + h* kv_3
			double kx_4[2] = {kx_1[0] + dt * k3[0], kx_1[1] + dt * k3[1]};

			// compute u(x_n + h kx_3, t_n+1)
			double k4[2] = {part_pos_old[0] + dt*kx_3[0], part_pos_old[1] + dt*kx_3[1]};
			device_hermite_interpolate_grad_2D(psi, k4, u, NX, NY, h, 1);
			// kv4 = F_v(x_n + h kx_3, v_n + h kv_3, t_n+1) = - (kx_4 - u(x_n+1 + h kx_3, t_n+1)) / tau_p
			k4[0] = - (kx_4[0] - u[0]) / tau_p;
			k4[1] = - (kx_4[1] - u[1]) / tau_p;

			// x_n+1 = x_n + dt/6 (kx_1 + 2*kx_2 + 2*kx_3 + kx_4)
			part_pos_old[0] += dt/6.0 * (kx_1[0] + 2*kx_2[0] + 2*kx_3[0] + kx_4[0]);
			part_pos_old[1] += dt/6.0 * (kx_1[1] + 2*kx_2[1] + 2*kx_3[1] + kx_4[1]);

			// v_n+1 = v_n + dt/6 (k1 + 2*k2 + 2*k3 + k4)
			particles_vel[2*i  ] += dt/6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
			particles_vel[2*i+1] += dt/6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);

			break;
		}
		// optimal runge kutta 3
		case 31: {
			double kx[6], kv[6];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			kx[0] = particles_vel[2*i  ]; kx[1] = particles_vel[2*i+1];
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, kv, NX, NY, h, 1);
			// kv_1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			kv[0] = - (kx[0] - kv[0])/tau_p; kv[1] = - (kx[1] - kv[1])/tau_p;

			// kx_2 = F_x(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			kx[2] = kx[0] + dt * kv[0]; kx[3] = kx[1] + dt * kv[1];
			// compute u(x_n + h kx_1, t_n+1)
			kv[2] = part_pos_old[0] + dt * kx[0]; kv[3] = part_pos_old[1] + dt * kx[1];
			device_hermite_interpolate_grad_2D(psi, kv+2, kv+2, NX, NY, h, 1);
			// kv2 = F_v(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = - (kx_2 - u(x_n + h/2kx_1, t_n+1/2)) / tau_p
			kv[2] = - (kx[2] - kv[2])/tau_p; kv[3] = - (kx[3] - kv[3])/tau_p;

			// kx_3 = F_x(x_n + h kx_1 - 2h kx_2, v_n + h kv_1 - 2h kv_2, t_n-1) = v_n + h* kv_1 -2h* kv_2
			kx[4] = kx[0] + dt * (kv[0] - 2*kv[2]); kx[5] = kx[1] + dt * (kv[1] - 2*kv[3]);
			// compute u(x_n + h kx_1 - 2h kx_2, t_n-1)
			kv[4] = part_pos_old[0] + dt*(kx[0]-2*kx[2]); kv[5] = part_pos_old[1] + dt*(kx[1]-2*kx[3]);
			device_hermite_interpolate_grad_2D(psi + 8*N, kv+4, kv+4, NX, NY, h, 1);
			// kv3 = F_v(x_n + h kx_1 - 2h kx_2, v_n + h kv_1 - 2h kv_2, t_n-1) = - (kx_3 - u(x_n+1 + h(kx_1-2kx_2), t_n-1)) / tau_p
			kv[4] = - (kx[4] - kv[4])/tau_p; kv[5] = - (kx[5] - kv[5])/tau_p;

			// x_n+1 = x_n + dt/12 (8*kx_1 + 5*kx_2 - kx_3)
			part_pos_old[0] += dt/12.0 * (8*kx[0] + 5*kx[2] - kx[4]);
			part_pos_old[1] += dt/12.0 * (8*kx[1] + 5*kx[3] - kx[5]);

			// v_n+1 = v_n + dt/12 (8*kv_1 + 5*kv_2 - kv_3)
			particles_vel[2*i  ] += dt/12.0 * (8*kv[0] + 5*kv[2] - kv[4]);
			particles_vel[2*i+1] += dt/12.0 * (8*kv[1] + 5*kv[3] - kv[5]);

			break;
		}
		// optimized runge kutta 4 with only one lagrange interpolation
		case 41: {
			double u[8], kx[8], kv[8];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			kx[0] = particles_vel[2*i  ]; kx[1] = particles_vel[2*i+1];
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*N, part_pos_old, kv, NX, NY, h, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			kv[0] = - (kx[0] - kv[0])/tau_p; kv[1] = - (kx[1] - kv[1])/tau_p;


			// kx_2 = F_x(x_n + h kx_1, v_n + h kv_1, t_n+1) = v_n + h kv_1
			kx[2] = kx[0] + dt * kv[0]; kx[3] = kx[1] + dt * kv[1];

			// compute u(x_n + h kx_1, t_n+1)
			kv[2] = part_pos_old[0] + dt * kx[0]; kv[3] = part_pos_old[1] + dt * kx[1];
			device_hermite_interpolate_grad_2D(psi, kv+2, kv+2, NX, NY, h, 1);
			// kv_2 = F_v(x_n + h kx_1, v_n + h kv_1, t_n+1) = - (kx_2 - u(x_n + hkx_1, t_n+1)) / tau_p
			kv[2] = - (kx[2] - kv[2])/tau_p; kv[3] = - (kx[3] - kv[3])/tau_p;


			// kx_3 = F_x(x_n + 3h/8 kx_1 + h/8 kx_2, v_n + 3h/8 kv_1 + h/8 kv_2, t_n+1/2) = v_n + + 3h/8 kv_1 + h/8 kv_2
			kx[4] = kx[0] + dt/8.0 * (3*kv[0] + kv[2]); kx[5] = kx[1] + dt/8.0 * (3*kv[1] + kv[3]);

			// compute u(x_n + 3h/8 kx_1 + h/8 kx_2, t_n+1/2) for all lagrange timesteps
			kv[4] = part_pos_old[0] + dt/8.0 * (3*kx[0] + kx[2]); kv[5] = part_pos_old[1] + dt/8.0 * (3*kx[1] + kx[3]);
			device_hermite_interpolate_grad_2D(psi, kv+4, u, NX, NY, h, 4);
			// kv_3 = F_v(x_n + 3h/8 kx_1 + h/8 kx_2, v_n + 3h/8 kv_1 + h/8 kv_2, t_n+1/2) = - (kx_3 - u(x_n + 3h/8 kx_1 + h/8 kx_2, t_n+1/2)) / tau_p with u_n+1/2 = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2
			kv[4] = - (kx[4] - (0.3125 * u[0] + 0.9375 * u[2] + -0.3125 * u[4] + 0.0625 * u[6])) / tau_p;
			kv[5] = - (kx[5] - (0.3125 * u[1] + 0.9375 * u[3] + -0.3125 * u[5] + 0.0625 * u[7])) / tau_p;


			// kx_4 = F_x(x_n + h/3 (-kx_2 + 4*kx_3), v_n + h/3 (-kv_2 + 4*kv_3), t_n+1) = v_n + h/3 (-kv_2 + 4*kv_3)
			kx[6] = kx[0] + dt/3.0 * (-kv[2] + 4*kv[4]); kx[7] = kx[1] + dt/3.0 * (-kv[3] + 4*kv[5]);

			// compute u(x_n + h/3 (-kx_2 + 4*kx_3), t_n+1)
			kv[6] = part_pos_old[0] + dt/3.0 * (-kx[2] + 4*kx[4]); kv[7] = part_pos_old[1] + dt/3.0 * (-kx[3] + 4*kx[5]);
			device_hermite_interpolate_grad_2D(psi, kv+6, kv+6, NX, NY, h, 1);
			// kv4 = F_v(x_n + h/3 (-kx_2 + 4*kx_3), v_n + h/3 (-kv_2 + 4*kv_3), t_n+1) = - (kx_4 - u(x_n + h/3 (-kx_2 + 4*kx_3), t_n+1)) / tau_p
			kv[6] = - (kx[6] - kv[6])/tau_p; kv[7] = - (kx[7] - kv[7])/tau_p;

			// x_n+1 = x_n + dt/12 (2*kx_1 - kx_2 + 8*kx_3 + 3*kx_4)
			part_pos_old[0] += dt/12.0 * (2*kx[0] - kx[2] + 8*kx[4] + 3*kx[6]);
			part_pos_old[1] += dt/12.0 * (2*kx[1] - kx[3] + 8*kx[5] + 3*kx[7]);

			// v_n+1 = v_n + dt/12 (2*kv_1 - kv_2 + 8*kv_3 + 3*kv_4)
			particles_vel[2*i  ] += dt/12.0 * (2*kv[0] - kv[2] + 8*kv[4] + 3*kv[6]);
			particles_vel[2*i+1] += dt/12.0 * (2*kv[1] - kv[3] + 8*kv[5] + 3*kv[7]);

			break;
		}
		// explicit midpoint rule from nicolas
		case 25: {
			double psi_x_p, psi_y_p; // Previous values of the x and y derivative of Psi.

		    device_hermite_interpolate_dx_dy_2D(psi + 4*N, part_pos_old[0], part_pos_old[1], &psi_x_p, &psi_y_p, NX, NY, h);

		    particles_vel[2 * i] =  particles_vel[2 * i] - 0.5*(dt/tau_p) * ((particles_vel[2 * i] - psi_y_p) + (particles_vel[2*i] + 0.5*dt*(particles_vel[2 * i] - psi_y_p)  - psi_y_p));
		    part_pos_old[0] += dt * particles_vel[2 * i];

		    particles_vel[2 * i + 1] =  particles_vel[2 * i + 1] - 0.5*(dt/tau_p) * (particles_vel[2 * i + 1] + psi_x_p + (particles_vel[2*i + 1 ] + 0.5*dt*(particles_vel[2 * i] + psi_x_p) + psi_x_p));
		    part_pos_old[1] += dt * particles_vel[2 * i + 1];
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
		    device_hermite_interpolate_dx_dy_2D(psi + 4*N, part_pos_old[0], part_pos_old[1], &psi_x_p, &psi_y_p, NX, NY, h);
		    u_p = psi_y_p; v_p = -psi_x_p;
		    k1_x = particles_vel[2*i] - u_p;
		    k1_y = particles_vel[2*i+1] - v_p;

		    // Compute k2
		    device_hermite_interpolate_dx_dy_2D(psi + 8*N, part_pos_old[0] + 0.5 * dt * k1_x, part_pos_old[1] + 0.5 * dt * k1_y, &psi_x_pp, &psi_y_pp, NX, NY, h);
		    u_pp = psi_y_pp; v_pp = -psi_x_pp;

		    device_hermite_interpolate_dx_dy_2D(psi + 4*N, part_pos_old[0] + 0.5 * dt * k1_x, part_pos_old[1] + 0.5 * dt * k1_y, &psi_x_p, &psi_y_p, NX, NY, h);
		    u_p = psi_y_p; v_p = -psi_x_p;

		    device_hermite_interpolate_dx_dy_2D(psi, part_pos_old[0] + 0.5 * dt * k1_x, part_pos_old[1] + 0.5 * dt * k1_y, &psi_x, &psi_y, NX, NY, h);
		    u = psi_y; v = -psi_x;
		    k2_x = particles_vel[2*i] + 0.5*dt*k1_x - (l[0] * u + l[1]* u_p + l[2] * u_pp);
		    k2_y = particles_vel[2*i+1] + 0.5*dt*k1_y - (l[0] * v + l[1] * v_p + l[2] * v_pp);

		    // Compute k3

		    device_hermite_interpolate_dx_dy_2D(psi, part_pos_old[0] - dt*k1_x + 2*dt*k2_x, part_pos_old[1] - dt*k1_y + 2*dt*k2_y , &psi_x, &psi_y, NX, NY, h);
		    u = psi_y; v = -psi_x;
		    k3_x = particles_vel[2*i]  - dt*k1_x + 2*dt*k2_x - u;
		    k3_y = particles_vel[2*i + 1] - dt*k1_y + 2*dt*k2_y - v;

		    particles_vel[2 * i] =  particles_vel[2 * i] - (dt/tau_p) * (k1_x + 4*k2_x + k3_x)/6;
		    part_pos_old[0] += dt * particles_vel[2*i];

		    particles_vel[2 * i + 1] = particles_vel[2 * i + 1] - (dt/tau_p) * (k1_y + 4*k2_y + k3_y)/6;
		    part_pos_old[1] += dt * particles_vel[2 * i + 1];
			break;
		}
		// default: zero
		default: { part_pos_old[0] = part_pos_old[1] = 0; particles_vel[2*i  ] = particles_vel[2*i+1] = 0; break; }
	}
	// map particle position back into domain, inspired by hermite version, this also now saves the values (only save)
	particles_pos[2*i]   = part_pos_old[0] - floor(part_pos_old[0]/LX)*LX;
	particles_pos[2*i+1] = part_pos_old[1] - floor(part_pos_old[1]/LY)*LX;

	// debugging of particle position
//	if (i == 0) printf("Particle number : %d - Position X : %f - Position Y : %f - Vel X : %f - Vel Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1], particles_vel[2*i], particles_vel[2*i+1]);

}
