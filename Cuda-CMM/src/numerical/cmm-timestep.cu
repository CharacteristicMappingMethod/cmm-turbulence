#include "cmm-timestep.h"

#include "../numerical/cmm-hermite.h"
#include "../grid/cudagrid2d.h"

#include "stdio.h"

/*******************************************************************
*		script with all different timestep methods
*******************************************************************/

__device__ void euler_exp(double *phi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order) {
	double u[8]; // velocity placeholders, set after largest l_order
	device_hermite_interpolate_grad_2D(phi, x_in, u, Grid, l_order);
	double k[2] = {d_L1[0] * u[0], d_L1[0] * u[1]};
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[0] += d_L1[i_l] * u[i_l*2  ];
		k[1] += d_L1[i_l] * u[i_l*2+1];
	}
	x_out[0] = x_in[0] - dt * k[0];
	x_out[1] = x_in[1] - dt * k[1];
}


// second order Heun
__device__ void RK2_heun(double *phi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order) {
	double u[8];  // velocity placeholders

	device_hermite_interpolate_grad_2D(phi, x_in, u, Grid, l_order);
	double k1[2] = {d_L1[0] * u[0], d_L1[0] * u[1]};
	for (int i_l = 1; i_l < l_order; i_l++) {
		k1[0] += d_L1[i_l] * u[i_l*2  ];
		k1[1] += d_L1[i_l] * u[i_l*2+1];
	}

	// k2 = u_tilde(x - dt k1, t_n)
	double k2[2] = {x_in[0] - dt*k1[0], x_in[1] - dt*k1[1]};
	device_hermite_interpolate_grad_2D(phi, k2, u, Grid, 1);
	// x_n+q = x_n - dt/2 (k1 + k2)
	x_out[0] = x_in[0] - dt * (k1[0] + u[0])/2;
	x_out[1] = x_in[1] - dt * (k1[1] + u[1])/2;
}


// second order adam bashfords with fixed point iteration to converge loop
__device__ void adam_bashford_2_pc(double *phi, double *x_in, double *x_out, TCudaGrid2D Grid, double dt) {
	double u[4];  // velocity placeholders
	double xf_p[2] = { x_in[0], x_in[1] };  // variables used for fixed-point iteration
	x_out[0] = x_in[0]; x_out[1] = x_in[1];  // initialize iteration

	// fixed point iteration for xf,yf using previous foot points (self-correction)
    #pragma unroll 10
	for(int ctr = 0; ctr<10; ctr++)
    {
		//step 1 on phi_p
        device_hermite_interpolate_grad_2D(phi + 4*Grid.N, xf_p, u + 2, Grid, 1);

        xf_p[0] = x_out[0] - dt * u[2];
        xf_p[1] = x_out[1] - dt * u[3];

		//step 2 on phi_p and phi
        device_hermite_interpolate_grad_2D(phi + 4*Grid.N, xf_p, u + 2, Grid, 1);
        device_hermite_interpolate_grad_2D(phi, x_out, u, Grid, 1);

        x_out[0] = x_in[0] - dt * (1.5*u[0] - 0.5*u[2]);
        x_out[1] = x_in[1] - dt * (1.5*u[1] - 0.5*u[3]);

    }
}


/*
 *  RKThree time step utilizing some intermediate steps
 *  Butcher tableau:
 *  1/2 | 1/2
 *   1  | -1   2
 *      | 1/6 2/3 1/6
 */
__device__ void RK3_classical(double *phi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order) {
	double u[8];  // velocity placeholders
	double k[6];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, Grid, l_order);
	// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
	k[0] = d_L1[0] * u[0]; k[1] = d_L1[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[0] += d_L1[i_l] * u[i_l*2  ];
		k[1] += d_L1[i_l] * u[i_l*2+1];
	};

	// compute u_tilde(X,t_n+1)
	k[2] = x_in[0] - dt*k[0]/2.0; k[3] = x_in[1] - dt*k[1]/2.0;
	device_hermite_interpolate_grad_2D(phi, k+2, u, Grid, l_order);
	// k1 = u_tilde(x,t_n+1) = 1.875*u_n - 1.25*u_n-1 + 0.375*u_n-2
	k[2] = d_L12[0] * u[0]; k[3] = d_L12[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[2] += d_L12[i_l] * u[i_l*2  ];
		k[3] += d_L12[i_l] * u[i_l*2+1];
	};

	//compute u_tilde(x + dt * k1 - 2*dt*k2, t_n+1 - dt)
	k[4] = x_in[0] + dt*k[0] - 2*dt*k[2]; k[5] = x_in[1] + dt*k[1] - 2*dt*k[3];
	device_hermite_interpolate_grad_2D(phi, k+4, k+4, Grid, 1);

	// k3 = u_tilde(x = k1 dt - 2 k2 dt, t_n) = u

	// build all RK-steps together
	x_out[0] = x_in[0] - dt * (k[0] + 4*k[2] + k[4])/6.0;
	x_out[1] = x_in[1] - dt * (k[1] + 4*k[3] + k[5])/6.0;
}


/*
 *  classical RKFour time step
 *  Butcher tableau:
 *  1/2 | 1/2
 *  1/2 |  0  1/2
 *   1  |  0   0   1
 *      | 1/6 1/3 1/3 1/6
 */
__device__ void RK4_classical(double *phi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order) {
	double u[8];  // velocity placeholders
	double k[8];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, Grid, l_order);
	// k1 = u_tilde(x,t_n+1) = 4*u_n - 6*u_n-1 + 4*u_n-2 - 1*u_n-3
	k[0] = d_L1[0] * u[0]; k[1] = d_L1[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[0] += d_L1[i_l] * u[i_l*2  ];
		k[1] += d_L1[i_l] * u[i_l*2+1];
	};


	// compute u_tilde(x - dt*k1/2, t_n+1/2)
	k[2] = x_in[0] - dt*k[0]/2.0; k[3] = x_in[1] - dt*k[1]/2.0;
	device_hermite_interpolate_grad_2D(phi, k+2, u, Grid, l_order);
	//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
	k[2] = d_L1[0] * u[0]; k[3] = d_L1[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[2] += d_L1[i_l] * u[i_l*2  ];
		k[3] += d_L1[i_l] * u[i_l*2+1];
	};

	// compute u_tilde(x - dt*k2/2, t_n+1/2)
	k[4] = x_in[0] - dt*k[2]/2.0; k[5] = x_in[1] - dt*k[3]/2.0;
	device_hermite_interpolate_grad_2D(phi, k+4, u, Grid, l_order);
	//k3 = u_tilde(x - k2 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
	k[4] = d_L1[0] * u[0]; k[5] = d_L1[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[4] += d_L1[i_l] * u[i_l*2  ];
		k[5] += d_L1[i_l] * u[i_l*2+1];
	};

	//compute u_tilde(x - dt*k3, t_n)
	k[6] = x_in[0] - dt*k[4]; k[7] = x_in[1] - dt*k[5];
	device_hermite_interpolate_grad_2D(phi, k+6, k+6, Grid, 1);

	// k4 = u_tilde(x - k3 dt, t_n) = u

	// build all RK-steps together
	x_out[0] = x_in[0] - dt * (k[0] + 2*k[2] + 2*k[4] + k[6])/6.0;
	x_out[1] = x_in[1] - dt * (k[1] + 2*k[3] + 2*k[5] + k[7])/6.0;
}


/*
 * Modified RKThree with negative times, faster than classical
 * Butcher tableau:
 * 1 |  1
 * 2 |  4   -2
 *   | 5/12 2/3 -1/12
 */
__device__ void RK3_optimized(double *phi, double *d_L1, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order) {
	double u[8];  // velocity placeholders
	double k[6];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, Grid, l_order);
	k[0] = d_L1[0] * u[0]; k[1] = d_L1[0] * u[1];
	// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[0] += d_L1[i_l] * u[i_l*2  ];
		k[1] += d_L1[i_l] * u[i_l*2+1];
	};

	// compute u_tilde(X - dt*k1, t_n)
	k[2] = x_in[0] - dt*k[0]; k[3] = x_in[1] - dt*k[1];
	device_hermite_interpolate_grad_2D(phi, k+2, k+2, Grid, 1);
	// k2 = u_tilde(X - dt*k1, t_n) = u_n

	// compute u_tilde(X - 4*dt*k1 + 2*dt*k2, t_n-1)
	k[4] = x_in[0] + 2*dt*(-2*k[0] + k[2]); k[5] = x_in[1] + 2*dt*(-2*k[1] + k[3]);
	device_hermite_interpolate_grad_2D(phi + 4*Grid.N, k+4, k+4, Grid, 1);
	// k3 = u_tilde(X - 4*dt*k1 + 2*dt*k2, t_n-1) = u_n-1

	// build all RK-steps together
	// x = x_n - dt (5*k1 + 8*k2 - k3)/12
	x_out[0] = x_in[0] - dt * (5*k[0] + 8*k[2] - k[4])/12.0;
	x_out[1] = x_in[1] - dt * (5*k[1] + 8*k[3] - k[5])/12.0;
}


/*
 *  modified RKFour time step of case IV with b=1/4 (Butcher (2016)), is faster than classical
 *  Butcher tableau:
 *   1 |   1
 *  1/2 | 3/8  1/8
 *   1  |  0  -1/3  4/3
 *      | 1/6 -1/12 2/3 1/4
 */
__device__ void RK4_optimized(double *phi, double *d_L1, double *d_L12, double *x_in, double *x_out, TCudaGrid2D Grid, double dt, int l_order) {
	double u[8];  // velocity placeholders
	double k[8];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, Grid, l_order);
	// k1 = u_tilde(X,t_n+1) = 4*u_n - 6*u_n-1 + 4*u_n-2 - 1*u_n-3
	k[0] = d_L1[0] * u[0]; k[1] = d_L1[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[0] += d_L1[i_l] * u[i_l*2  ];
		k[1] += d_L1[i_l] * u[i_l*2+1];
	};

	// compute u_tilde(X - dt*k1, t_n)
	k[2] = x_in[0] - dt*k[0]; k[3] = x_in[1] - dt*k[1];
	device_hermite_interpolate_grad_2D(phi, k+2, k+2, Grid, 1);
	// k2 = u_tilde(X - dt*k1, t_n) = u_n

	// compute u_tilde(X - 3/8 dt*k1 - 1/8 dt*k2, t_n+1/2)
	k[4] = x_in[0] - dt*(3*k[0] + k[2])/8.0; k[5] = x_in[1] - dt*(3*k[1] + k[3])/8.0;
	device_hermite_interpolate_grad_2D(phi, k+4, u, Grid, l_order);
	//k3 = u_tilde(X - 3/8 dt*k1 - 1/8 dt*k2, t_n+1/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
	k[4] = d_L1[0] * u[0]; k[5] = d_L1[0] * u[1];
	for (int i_l = 1; i_l < l_order; i_l++) {
		k[4] += d_L1[i_l] * u[i_l*2  ];
		k[5] += d_L1[i_l] * u[i_l*2+1];
	};

	//compute u_tilde(X + 1/3 dt*k2 - 4/3 dt*k3, t_n)
	k[6] = x_in[0] + dt*(k[2] - 4*k[4])/3.0; k[7] = x_in[1] + dt*(k[3] - 4*k[5])/3.0;
	device_hermite_interpolate_grad_2D(phi, k+6, k+6, Grid, 1);
	// k3 = u_tilde(X + 1/3 dt*k2 - 4/3 dt*k3, t_n) = u

	// build all RK-steps together
	// x = x_n - dt (2*k1 - k2 + 8*k3 + 3*k4)/12
	x_out[0] = x_in[0] - dt * (2*k[0] - k[2] + 8*k[4] + 3*k[6])/12.0;
	x_out[1] = x_in[1] - dt * (2*k[1] - k[3] + 8*k[5] + 3*k[7])/12.0;
}



__host__ __device__ double get_L_coefficient(double *t, double t_next, int loop_ctr, int i_point, int l_order) {
	double coeff = 1;
	for (int i_p = 0; i_p < l_order; ++i_p) {
		if (i_p != i_point) {
			coeff *= (t_next - t[loop_ctr-i_p]) / (t[loop_ctr-i_point] - t[loop_ctr-i_p]);
		}
	}
	return coeff;
}
