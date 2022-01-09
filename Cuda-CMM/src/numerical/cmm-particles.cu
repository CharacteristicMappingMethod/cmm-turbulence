#include "cmm-particles.h"

#include "../grid/cmm-grid2d.h"
#include "../numerical/cmm-timestep.h"
#include "stdio.h"
#include "../ui/settings.h"

/*******************************************************************
*						  Particule advection        			   *
*******************************************************************/

__constant__ double d_Lp12[4];

__global__ void k_rescale(int Nb_particle, double particles_center_x, double particles_center_y, double particles_width_x, double particles_width_y, double *particles_pos, double LX, double LY) {  // Rescaling the particles uniform distribution on the square [0,s]x[0,s] (s = twoPI)

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double part_pos_old[2];
	part_pos_old[0] = particles_center_x + particles_width_x * (particles_pos[2*i  ] - 0.5);
	part_pos_old[1] = particles_center_y + particles_width_y * (particles_pos[2*i+1] - 0.5);

	particles_pos[2*i]   = part_pos_old[0] - floor(part_pos_old[0]/LX)*LX;
	particles_pos[2*i+1] = part_pos_old[1] - floor(part_pos_old[1]/LY)*LX;
}


// initialize inertial particles with fluid velocity
__global__ void Particle_advect_inertia_init(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi, TCudaGrid2D Grid){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	// compute velocity and transcribe directly
	device_hermite_interpolate_grad_2D(psi, particles_pos + 2*i, particles_vel + 2*i, Grid, 1);
}


void particles_advect(SettingsCMM SettingsMain, TCudaGrid2D Grid_psi, double *particles_pos, double *particles_vel,
		double *psi, double *t, double *dt, int loop_ctr, int particles_block, int particles_thread) {
	// compute lagrange coefficients from dt vector for timesteps n+dt and n+dt/2, this makes them dynamic
	double h_L12[4];  // constant memory for lagrange coefficient to be computed only once
	int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;  // dt and t are shifted because of initial previous steps
	for (int i_p = 0; i_p < SettingsMain.getLagrangeOrder(); ++i_p) {
		// position loop_ctr_l+1, because velocity was already computed
		h_L12[i_p] = get_L_coefficient(t, t[loop_ctr_l] + dt[loop_ctr_l+1]/2.0, loop_ctr_l+1, i_p, SettingsMain.getLagrangeOrder());
	}

	// copy to constant memory
	cudaMemcpyToSymbol(d_Lp12, h_L12, sizeof(double)*4);

	Particle_advect<<<particles_block, particles_thread>>>(SettingsMain.getParticlesNum(), dt[loop_ctr_l + 1], particles_pos, psi,
			Grid_psi, SettingsMain.getParticlesTimeIntegrationNum(), SettingsMain.getLagrangeOrder());
	// loop for all tau p
	for(int i_tau_p = 1; i_tau_p < SettingsMain.getParticlesTauNum(); i_tau_p++){
		Particle_advect_inertia<<<particles_block, particles_thread>>>(SettingsMain.getParticlesNum(), dt[loop_ctr_l + 1],
				particles_pos + 2*SettingsMain.getParticlesNum()*i_tau_p, particles_vel + 2*SettingsMain.getParticlesNum()*i_tau_p, psi,
				Grid_psi, SettingsMain.particles_tau[i_tau_p], SettingsMain.getParticlesTimeIntegrationNum(), SettingsMain.getLagrangeOrder());
	}
}


// Advect particles using Hermite interpolation and time schemes.

__global__ void Particle_advect(int Nb_particle, double dt, double *particles_pos, double *psi,
		TCudaGrid2D Grid, int particles_time_integration_num, int l_order){

	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = Grid.NX*Grid.hx;
	double LY = Grid.NY*Grid.hy;

	// reading globally is expensive, so we only want to do it once
	double part_pos_old[2] = { particles_pos[2*i  ], particles_pos[2*i+1] };

	// compute new position, important: psi is set at t_n+1, psi_previous at t_n and pri_previous_p at t_n-1
	switch (particles_time_integration_num) {
		case 10:  // euler explicit
		{
			double u[2];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, u, Grid, 1);

			// x_n+1 = x_n + dt*u_n
			part_pos_old[0] += dt * u[0];
			part_pos_old[1] += dt * u[1];
			break;
		}
		case 20:  // heun / modified euler / trapezoidal method
		{
			double k[4];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, k, Grid, 1);

			// k2 = f_x(x_n + h*k1, t_n ) = u_n+1 ( x_n + h*k1 )
			k[2] = part_pos_old[0] + dt*k[0]; k[3] = part_pos_old[1] + dt*k[1];
			device_hermite_interpolate_grad_2D(psi, k+2, k+2, Grid, 1);

			// x_n+1 = x_n + dt/2 (k1 + k2)
			part_pos_old[0] += dt * (k[0] + k[2])/2;
			part_pos_old[1] += dt * (k[1] + k[3])/2;
			break;
		}
		case 30:  // runge kutta 3 for fluid particles, the nicolas version is identical
		{
			double u[8], k[6];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, k, Grid, 1);

			// k2 = f_x(x_n + h/2*k1, t_n+1/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1 ( x_n + h/2*k1 )
			k[2] = part_pos_old[0] + dt*k[0]/2.0; k[3] = part_pos_old[1] + dt*k[1]/2.0;
			device_hermite_interpolate_grad_2D(psi, k+2, u, Grid, l_order);
			k[2] = d_Lp12[0] * u[0]; k[3] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k[2] += d_Lp12[i_l] * u[i_l*2  ];
				k[3] += d_Lp12[i_l] * u[i_l*2+1];
			}

			// k3 = f_x(x_n - h*k1 + 2h*k2, t_n+1) = u_n+1 ( x_n - h*k1 + 2h*k2, t_n+1 )
			k[4] = part_pos_old[0] - dt*k[0] + 2*dt*k[2]; k[5] = part_pos_old[1] - dt*k[1] + 2*dt*k[3];
			device_hermite_interpolate_grad_2D(psi, k+4, k+4, Grid, 1);

			// x_n+q = x_n + dt/6 (k1 + 4*k2 + k3)
			part_pos_old[0] += dt * (k[0] + 4*k[2] + k[4]) / 6.0;
			part_pos_old[1] += dt * (k[1] + 4*k[3] + k[5]) / 6.0;

			break;
		}
		case 40:  // runge kutta 4 for fluid particles
		{
			double u[8], k[8];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, k, Grid, 1);

			// k2 = f_x(x_n + h/2*k1, t_n+1/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1 ( x_n + h/2*k1 )
			k[2] = part_pos_old[0] + dt*k[0]/2.0; k[3] = part_pos_old[1] + dt*k[1]/2.0;
			device_hermite_interpolate_grad_2D(psi, k+2, u, Grid, l_order);
			k[2] = d_Lp12[0] * u[0]; k[3] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k[2] += d_Lp12[i_l] * u[i_l*2  ];
				k[3] += d_Lp12[i_l] * u[i_l*2+1];
			}

			// k2 = f_x(x_n + h/2*k2, t_n+1/2) = 0.375*u_n+1 + 0.75*u_n -0.125*u_n-1 ( x_n + h/2*k2 )
			k[4] = part_pos_old[0] + dt*k[2]/2.0; k[5] = part_pos_old[1] + dt*k[3]/2.0;
			device_hermite_interpolate_grad_2D(psi, k+4, u, Grid, l_order);
			k[4] = d_Lp12[0] * u[0]; k[5] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k[4] += d_Lp12[i_l] * u[i_l*2  ];
				k[5] += d_Lp12[i_l] * u[i_l*2+1];
			}

			// k4 = f_x(x_n + h*k3, t_n ) = u_n+1 ( x_n + h*k3 )
			k[6] = part_pos_old[0] + dt*k[4]; k[7] = part_pos_old[1] + dt*k[5];
			device_hermite_interpolate_grad_2D(psi, k+6, k+6, Grid, 1);

			// x_n+1 = x_n + dt/6 (k1 + 2*k2 + 2*k3 + k4)
			part_pos_old[0] += dt * (k[0] + 2*k[2] + 2*k[4] + k[6]) / 6.0;
			part_pos_old[1] += dt * (k[1] + 2*k[3] + 2*k[5] + k[7]) / 6.0;

			break;
		}
		case 31:  // modified rkthree with optimal timesteps chosen to result in no lagrange interpolation
		{
			double k[6];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, k, Grid, 1);

			// k2 = f_x(x_n + h*k1, t_n ) = u_n+1 ( x_n + h*k1 )
			k[2] = part_pos_old[0] + dt*k[0]; k[3] = part_pos_old[1] + dt*k[1];
			device_hermite_interpolate_grad_2D(psi, k+2, k+2, Grid, 1);

			// k3 = f_x(x_n + h*k1 - 2h*k2, t_n-1) = u_n-1 ( x_n + h*k1 - 2h*k2 )
			k[4] = part_pos_old[0] + dt*(k[0] - 2*k[2]); k[5] = part_pos_old[1] + dt*(k[1] - 2*k[3]);
			device_hermite_interpolate_grad_2D(psi + 8*Grid.N, k+4, k+4, Grid, 1);

			// x_n+1 = x_n + dt/12 (8*k1 + 5*k2 - k3)
			part_pos_old[0] += dt * (8*k[0] + 5*k[2] - k[4])/12.0;
			part_pos_old[1] += dt * (8*k[1] + 5*k[3] - k[5])/12.0;
			break;
		}
		case 41:  // modified runge kutta 4 for fluid particles with optimal timesteps chosen to result in only ine lagrange interpolation
		{
			double u[8], k[8];
			// k1 = f_x(x_n, t_n ) = u_n ( x_n )
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, k, Grid, 1);

			// k2 = f_x(x_n + h*k1, t_n ) = u_n+1 ( x_n + h*k1 )
			k[2] = part_pos_old[0] + dt*k[0]; k[3] = part_pos_old[1] + dt*k[1];
			device_hermite_interpolate_grad_2D(psi, k+2, k+2, Grid, 1);

			// k3 = f_x(x_n + 3h/8*k1 + h/8*k2, t_n+1/2) = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2 ( x_n + 3h/8*k1 + h/8*k2 )
			k[4] = part_pos_old[0] + dt/8.0*(3*k[0] + k[2]); k[5] = part_pos_old[1] + dt/8.0*(3*k[1] + k[3]);
			device_hermite_interpolate_grad_2D(psi, k+4, u, Grid, l_order);
			k[4] = d_Lp12[0] * u[0]; k[5] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k[4] += d_Lp12[i_l] * u[i_l*2  ];
				k[5] += d_Lp12[i_l] * u[i_l*2+1];
			}

			// k4 = f_x(x_n - h/3*k2 + 4h/3 k3, t_n+1) = u_n+1 ( x_n - h/3*k2 + 4h/3 k3 )
			k[6] = part_pos_old[0] + dt/3.0*(-k[2] + 4*k[4]); k[7] = part_pos_old[1] + dt/3.0*(-k[3] + 4*k[5]);
			device_hermite_interpolate_grad_2D(psi, k+6, k+6, Grid, 1);

			// x_n+1 = x_n + dt/12 (2*k1 - k2 + 8*k3 + 3*k4)
			part_pos_old[0] += dt * (2*k[0] - k[2] + 8*k[4] + 3*k[6])/12.0;
			part_pos_old[1] += dt * (2*k[1] - k[3] + 8*k[5] + 3*k[7])/12.0;

			break;
		}
		// default: zero
		default: { part_pos_old[0] = part_pos_old[1] = 0; break; }
	}
	// map particle position back into domain, inspired by hermite version
	particles_pos[2*i]   = part_pos_old[0] - floor(part_pos_old[0]/LX)*LX;
	particles_pos[2*i+1] = part_pos_old[1] - floor(part_pos_old[1]/LY)*LX;

	// debugging of particle position
//	if (i < 1) printf("Particle number : %d - Position X : %f - Position Y : %f - \n", i, particles_pos[2*i], particles_pos[2*i+1]);
}


// advection of inertial particles
__global__ void Particle_advect_inertia(int Nb_particle, double dt, double *particles_pos, double *particles_vel, double *psi,
		TCudaGrid2D Grid, double tau_p, int particles_time_integration_num, int l_order) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= Nb_particle)
		return;

	double LX = Grid.NX*Grid.hx;
	double LY = Grid.NY*Grid.hy;

	// reading globally is expensive, so we only want to do it once
	double part_pos_old[2] = { particles_pos[2*i  ], particles_pos[2*i+1] };

	// compute new velocity, important: psi is set at t_n+1, psi_previous at t_n ...
	switch (particles_time_integration_num) {
		case 10:  // euler exp
		{
			double u[2];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = { particles_vel[2*i  ], particles_vel[2*i+1] };
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, u, Grid, 1);

			// x_n+1 = x_n + dt v_n
			part_pos_old[0] += dt * kx_1[0];
			part_pos_old[1] += dt * kx_1[1];

			// kv1 = F_v(v_n, t_n) = - (v_n - u_n(x_n)) / tau_p
			// v_n+1 = v_n + dt*kv1
			particles_vel[2*i  ] -= dt * (kx_1[0] - u[0])/tau_p;
			particles_vel[2*i+1] -= dt * (kx_1[1] - u[1])/tau_p;
			break;
		}
		case 20:  // heun / modified euler / trapezoidal method
		{
			double u[4];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, u+2, Grid, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			double k1[2] = {- (kx_1[0] - u[2])/tau_p, - (kx_1[1] - u[3])/tau_p};

			// kx_2 = F_x(x_n + h kx_1, v_n + h kv_1, t_n+1) = v_n + h kv_1
			double kx_2[2] = {kx_1[0] + dt * k1[0], kx_1[1] + dt * k1[1]};

			// compute u(x_n + h kx_1, t_n+1)
			double k2[2] = {part_pos_old[0] + dt * kx_1[0], part_pos_old[1] + dt * kx_1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, Grid, 1);
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
		case 30:  // classical runge kutta 3
		{
			double u[8];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, u+2, Grid, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			double k1[2] = {- (kx_1[0] - u[2])/tau_p, - (kx_1[1] - u[3])/tau_p};

			// kx_2 = F_x(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			double kx_2[2] = {kx_1[0] + dt/2.0 * k1[0], kx_1[1] + dt/2.0 * k1[1]};

			// compute u(x_n + h/2 kx_1, t_n+1/2) for all lagrange timesteps
			double k2[2] = {part_pos_old[0] + dt/2.0 * kx_1[0], part_pos_old[1] + dt/2.0 * kx_1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, Grid, l_order);
			k2[0] = d_Lp12[0] * u[0], k2[1] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k2[0] += d_Lp12[i_l] * u[i_l*2  ];
				k2[1] += d_Lp12[i_l] * u[i_l*2+1];
			}
			// kv2 = F_v(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = - (kx_2 - u(x_n + h/2kx_1, t_n+1/2)) / tau_p with u_n+1/2 = 3/8*u_n+1 + 3/4*u_n - 1/8*u_n-1
			k2[0] = - (kx_2[0] - k2[0]) / tau_p;
			k2[1] = - (kx_2[1] - k2[1]) / tau_p;

			// kx_3 = F_x(x_n - h kx_1 + 2h kx_2, v_n - h kv_1 + 2h kv_2, t_n+1) = v_n -h* kv_1 + 2h* kv_2
			double kx_3[2] = {kx_1[0] + dt * (-k1[0] + 2*k2[0]), kx_1[1] + dt * (-k1[1] + 2*k2[1])};

			// compute u(x_n - h kx_1 + 2h kx_2, t_n+1)
			double k3[2] = {part_pos_old[0] + dt*(-kx_1[0]+2*kx_2[0]), part_pos_old[1] + dt*(-kx_1[1]+2*kx_2[1])};
			device_hermite_interpolate_grad_2D(psi, k3, u, Grid, 1);
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
		case 40:  // classical runge kutta 4
		{
			double u[8];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			double kx_1[2] = {particles_vel[2*i  ], particles_vel[2*i+1]};
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, u+2, Grid, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			double k1[2] = {- (kx_1[0] - u[2])/tau_p, - (kx_1[1] - u[3])/tau_p};


			// kx_2 = F_x(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			double kx_2[2] = {kx_1[0] + dt/2.0 * k1[0], kx_1[1] + dt/2.0 * k1[1]};

			// compute u(x_n + h/2 kx_1, t_n+1/2) for all lagrange timesteps
			double k2[2] = {part_pos_old[0] + dt/2.0 * kx_1[0], part_pos_old[1] + dt/2.0 * kx_1[1]};
			device_hermite_interpolate_grad_2D(psi, k2, u, Grid, l_order);
			k2[0] = d_Lp12[0] * u[0], k2[1] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k2[0] += d_Lp12[i_l] * u[i_l*2  ];
				k2[1] += d_Lp12[i_l] * u[i_l*2+1];
			}
			// kv2 = F_v(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = - (kx_2 - u(x_n + h/2kx_1, t_n+1/2)) / tau_p with u_n+1/2 = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2
			k2[0] = - (kx_2[0] - k2[0]) / tau_p;
			k2[1] = - (kx_2[1] - k2[1]) / tau_p;


			// kx_3 = F_x(x_n + h/2 kx_2, v_n + h/2 kv_2, t_n+1/2) = v_n + h/2 kv_2
			double kx_3[2] = {kx_1[0] + dt/2.0 * k2[0], kx_1[1] + dt/2.0 * k2[1]};

			// compute u(x_n + h/2 kx_2, t_n+1/2) for all lagrange timesteps
			double k3[2] = {part_pos_old[0] + dt/2.0 * kx_2[0], part_pos_old[1] + dt/2.0 * kx_2[1]};
			device_hermite_interpolate_grad_2D(psi, k3, u, Grid, l_order);
			k3[0] = d_Lp12[0] * u[0], k3[1] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				k3[0] += d_Lp12[i_l] * u[i_l*2  ];
				k3[1] += d_Lp12[i_l] * u[i_l*2+1];
			}
			// kv3 = F_v(x_n + h/2 kx_2, v_n + h/2 kv_2, t_n+1/2) = - (kx_3 - u(x_n + h/2kx_2, t_n+1/2)) / tau_p with u_n+1/2 = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2
			k3[0] = - (kx_3[0] - k3[0]) / tau_p;
			k3[1] = - (kx_3[1] - k3[1]) / tau_p;


			// kx_4 = F_x(x_n + h kx_3, v_n + h kv_3, t_n+1) = v_n + h* kv_3
			double kx_4[2] = {kx_1[0] + dt * k3[0], kx_1[1] + dt * k3[1]};

			// compute u(x_n + h kx_3, t_n+1)
			double k4[2] = {part_pos_old[0] + dt*kx_3[0], part_pos_old[1] + dt*kx_3[1]};
			device_hermite_interpolate_grad_2D(psi, k4, u, Grid, 1);
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
		case 31:  // optimal runge kutta 3
		{
			double kx[6], kv[6];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			kx[0] = particles_vel[2*i  ]; kx[1] = particles_vel[2*i+1];
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, kv, Grid, 1);
			// kv_1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			kv[0] = - (kx[0] - kv[0])/tau_p; kv[1] = - (kx[1] - kv[1])/tau_p;

			// kx_2 = F_x(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = v_n + h/2 kv_1
			kx[2] = kx[0] + dt * kv[0]; kx[3] = kx[1] + dt * kv[1];
			// compute u(x_n + h kx_1, t_n+1)
			kv[2] = part_pos_old[0] + dt * kx[0]; kv[3] = part_pos_old[1] + dt * kx[1];
			device_hermite_interpolate_grad_2D(psi, kv+2, kv+2, Grid, 1);
			// kv2 = F_v(x_n + h/2 kx_1, v_n + h/2 kv_1, t_n+1/2) = - (kx_2 - u(x_n + h/2kx_1, t_n+1/2)) / tau_p
			kv[2] = - (kx[2] - kv[2])/tau_p; kv[3] = - (kx[3] - kv[3])/tau_p;

			// kx_3 = F_x(x_n + h kx_1 - 2h kx_2, v_n + h kv_1 - 2h kv_2, t_n-1) = v_n + h* kv_1 -2h* kv_2
			kx[4] = kx[0] + dt * (kv[0] - 2*kv[2]); kx[5] = kx[1] + dt * (kv[1] - 2*kv[3]);
			// compute u(x_n + h kx_1 - 2h kx_2, t_n-1)
			kv[4] = part_pos_old[0] + dt*(kx[0]-2*kx[2]); kv[5] = part_pos_old[1] + dt*(kx[1]-2*kx[3]);
			device_hermite_interpolate_grad_2D(psi + 8*Grid.N, kv+4, kv+4, Grid, 1);
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
		case 41:  // optimized runge kutta 4 with only one lagrange interpolation
		{
			double u[8], kx[8], kv[8];
			// kx_1 = F_x(x_n, v_n, t_n) = v_n
			kx[0] = particles_vel[2*i  ]; kx[1] = particles_vel[2*i+1];
			// compute u(x_n,t_n)
			device_hermite_interpolate_grad_2D(psi + 4*Grid.N, part_pos_old, kv, Grid, 1);
			// kv1 = F_v(x_n, v_n, t_n) = - (kx_1 - u(x_n, t_n)) / tau_p
			kv[0] = - (kx[0] - kv[0])/tau_p; kv[1] = - (kx[1] - kv[1])/tau_p;


			// kx_2 = F_x(x_n + h kx_1, v_n + h kv_1, t_n+1) = v_n + h kv_1
			kx[2] = kx[0] + dt * kv[0]; kx[3] = kx[1] + dt * kv[1];

			// compute u(x_n + h kx_1, t_n+1)
			kv[2] = part_pos_old[0] + dt * kx[0]; kv[3] = part_pos_old[1] + dt * kx[1];
			device_hermite_interpolate_grad_2D(psi, kv+2, kv+2, Grid, 1);
			// kv_2 = F_v(x_n + h kx_1, v_n + h kv_1, t_n+1) = - (kx_2 - u(x_n + hkx_1, t_n+1)) / tau_p
			kv[2] = - (kx[2] - kv[2])/tau_p; kv[3] = - (kx[3] - kv[3])/tau_p;


			// kx_3 = F_x(x_n + 3h/8 kx_1 + h/8 kx_2, v_n + 3h/8 kv_1 + h/8 kv_2, t_n+1/2) = v_n + + 3h/8 kv_1 + h/8 kv_2
			kx[4] = kx[0] + dt/8.0 * (3*kv[0] + kv[2]); kx[5] = kx[1] + dt/8.0 * (3*kv[1] + kv[3]);

			// compute u(x_n + 3h/8 kx_1 + h/8 kx_2, t_n+1/2) for all lagrange timesteps
			kv[4] = part_pos_old[0] + dt/8.0 * (3*kx[0] + kx[2]); kv[5] = part_pos_old[1] + dt/8.0 * (3*kx[1] + kx[3]);
			device_hermite_interpolate_grad_2D(psi, kv+4, u, Grid, l_order);
			kv[4] = d_Lp12[0] * u[0], kv[5] = d_Lp12[0] * u[1];
			for (int i_l = 1; i_l < l_order; i_l++) {
				kv[4] += d_Lp12[i_l] * u[i_l*2  ];
				kv[5] += d_Lp12[i_l] * u[i_l*2+1];
			}
			// kv_3 = F_v(x_n + 3h/8 kx_1 + h/8 kx_2, v_n + 3h/8 kv_1 + h/8 kv_2, t_n+1/2) = - (kx_3 - u(x_n + 3h/8 kx_1 + h/8 kx_2, t_n+1/2)) / tau_p with u_n+1/2 = 0.3125*u_n+1 + 0.9375*u_n - 0.3125*u_n-1 + 0.0625*u_n-2
			kv[4] = - (kx[4] - kv[4]) / tau_p;
			kv[5] = - (kx[5] - kv[5]) / tau_p;


			// kx_4 = F_x(x_n + h/3 (-kx_2 + 4*kx_3), v_n + h/3 (-kv_2 + 4*kv_3), t_n+1) = v_n + h/3 (-kv_2 + 4*kv_3)
			kx[6] = kx[0] + dt/3.0 * (-kv[2] + 4*kv[4]); kx[7] = kx[1] + dt/3.0 * (-kv[3] + 4*kv[5]);

			// compute u(x_n + h/3 (-kx_2 + 4*kx_3), t_n+1)
			kv[6] = part_pos_old[0] + dt/3.0 * (-kx[2] + 4*kx[4]); kv[7] = part_pos_old[1] + dt/3.0 * (-kx[3] + 4*kx[5]);
			device_hermite_interpolate_grad_2D(psi, kv+6, kv+6, Grid, 1);
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
		// default: zero
		default: { part_pos_old[0] = part_pos_old[1] = 0; particles_vel[2*i  ] = particles_vel[2*i+1] = 0; break; }
	}
	// map particle position back into domain, inspired by hermite version, this also now saves the values (only save)
	particles_pos[2*i]   = part_pos_old[0] - floor(part_pos_old[0]/LX)*LX;
	particles_pos[2*i+1] = part_pos_old[1] - floor(part_pos_old[1]/LY)*LX;

	// debugging of particle position
//	if (i == 0) printf("Particle number : %d - Position X : %f - Position Y : %f - Vel X : %f - Vel Y : %f\n", i, particles_pos[2*i], particles_pos[2*i+1], particles_vel[2*i], particles_vel[2*i+1]);

}
