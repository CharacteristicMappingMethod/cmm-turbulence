#include "cmm-timestep.h"

/*******************************************************************
*		script with all different timestep methods
*******************************************************************/

__device__ void euler_exp(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	double u[2]; // velocity placeholders
	device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 1);
	x_out[0] = x_in[0] - dt * u[0];
	x_out[1] = x_in[1] - dt * u[1];
}


// second order Heun
__device__ void RK2_heun(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	long int N = NX * NY;
	double u[4];  // velocity placeholders

	double k1[2];
	// k1 = u_tilde(x, t_n+1) = 2*u_n - 1*u_n-1
	device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 2);
	k1[0] = 2*u[0] - u[2];
    k1[1] = 2*u[1] - u[3];

	// k2 = u_tilde(x - dt k1, t_n)
	double k2[2] = {x_in[0] - dt*k1[0], x_in[1] - dt*k1[1]};
	device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 1);
	// x_n+q = x_n - dt/2 (k1 + k2)
	x_out[0] = x_in[0] - dt * (k1[0] + u[0])/2;
	x_out[1] = x_in[1] - dt * (k1[1] + u[1])/2;
}


// second order adam bashfords with fixed point iteration to converge loop
__device__ void adam_bashford_2_pc(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	long int N = NX * NY;
	double u[4];  // velocity placeholders
	double xf_p[2] = { x_in[0], x_in[1] };  // variables used for fixed-point iteration
	x_out[0] = x_in[0]; x_out[1] = x_in[1];  // initialize iteration

	// fixed point iteration for xf,yf using previous foot points (self-correction)
    #pragma unroll 10
	for(int ctr = 0; ctr<10; ctr++)
    {
		//step 1 on phi_p
        device_hermite_interpolate_grad_2D(phi + 4*N, xf_p, u + 2, NX, NY, h, 1);

        xf_p[0] = x_out[0] - dt * u[2];
        xf_p[1] = x_out[1] - dt * u[3];

		//step 2 on phi_p and phi
        device_hermite_interpolate_grad_2D(phi + 4*N, xf_p, u + 2, NX, NY, h, 1);
        device_hermite_interpolate_grad_2D(phi, x_out, u, NX, NY, h, 1);

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
__device__ void RK3_classical(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	double u[6];  // velocity placeholders
	double k1[2], k2[2], k3[2];  // step placeholders

	// a test with different lagrange interpolations
//	int lagrange_order = 1;
//	double u[8];  // velocity placeholders
//	switch (lagrange_order) {
//		case 1: {
//			// compute u_tilde(X,t_n+1)
//			device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 1);
//
//			// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
//			k1[0] = u[0];
//			k1[1] = u[1];
//
//			// compute u_tilde(x - dt*k1/2, t_n+1 - dt/2)
//			k2[0] = x_in[0] - dt*k1[0]/2.0; k2[1] = x_in[1] - dt*k1[1]/2.0;
//			device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 1);
//
//			//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 1.875*u_n - 1.25*u_n-1 + 0.375*u_n-2
//			k2[0] = u[0];
//			k2[1] = u[1];
//			break;
//		}
//		case 2: {
//			// compute u_tilde(X,t_n+1)
//			device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 2);
//
//			// k1 = u_tilde(X,t_n+1) = 4*u_n - 6*u_n-1 + 4*u_n-2 - 1*u_n-3
//			k1[0] = 2 * u[0] - u[2];
//			k1[1] = 2 * u[1] - u[3];
//
//			// compute u_tilde(x - dt*k1/2, t_n+1/2)
//			k2[0] = x_in[0] - dt*k1[0]/2.0; k2[1] = x_in[1] - dt*k1[1]/2.0;
//			device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 2);
//
//			//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
//			k2[0] = 1.5 * u[0] + -0.5 * u[2];
//			k2[1] = 1.5 * u[1] + -0.5 * u[3];
//			break;
//		}
//		case 3: {
			// compute u_tilde(X,t_n+1)
			device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 3);

			// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
			k1[0] = 3 * u[0] + -3 * u[2] + u[4];
			k1[1] = 3 * u[1] + -3 * u[3] + u[5];

			// compute u_tilde(x - dt*k1/2, t_n+1 - dt/2)
			k2[0] = x_in[0] - dt*k1[0]/2.0; k2[1] = x_in[1] - dt*k1[1]/2.0;
			device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 3);

			//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 1.875*u_n - 1.25*u_n-1 + 0.375*u_n-2
			k2[0] = 1.875 * u[0] + -1.25 * u[2] + 0.375 * u[4];
			k2[1] = 1.875 * u[1] + -1.25 * u[3] + 0.375 * u[5];
//			break;
//		}
//		case 4: {
//			// compute u_tilde(X,t_n+1)
//			device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 4);
//
//			// k1 = u_tilde(X,t_n+1) = 4*u_n - 6*u_n-1 + 4*u_n-2 - 1*u_n-3
//			k1[0] = 4 * u[0] + -6 * u[2] + 4 * u[4] + -1 * u[6];
//			k1[1] = 4 * u[1] + -6 * u[3] + 4 * u[5] + -1 * u[7];
//
//			// compute u_tilde(x - dt*k1/2, t_n+1/2)
//			k2[0] = x_in[0] - dt*k1[0]/2.0; k2[1] = x_in[1] - dt*k1[1]/2.0;
//			device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 4);
//
//			//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
//			k2[0] = 2.1875 * u[0] + -2.1875 * u[2] + 1.3125 * u[4] + -0.3125 * u[6];
//			k2[1] = 2.1875 * u[1] + -2.1875 * u[3] + 1.3125 * u[5] + -0.3125 * u[7];
//			break;
//		}
//	}



	//compute u_tilde(x + dt * k1 - 2*dt*k2, t_n+1 - dt)
	k3[0] = x_in[0] + dt*k1[0] - 2*dt*k2[0]; k3[1] = x_in[1] + dt*k1[1] - 2*dt*k2[1];
	device_hermite_interpolate_grad_2D(phi, k3, u, NX, NY, h, 1);

	// k3 = u_tilde(x = k1 dt - 2 k2 dt, t_n) = u

	// build all RK-steps together
	x_out[0] = x_in[0] - dt * (k1[0] + 4*k2[0] + u[0])/6.0;
	x_out[1] = x_in[1] - dt * (k1[1] + 4*k2[1] + u[1])/6.0;
}


/*
 *  classical RKFour time step
 *  Butcher tableau:
 *  1/2 | 1/2
 *  1/2 |  0  1/2
 *   1  |  0   0   1
 *      | 1/6 1/3 1/3 1/6
 */
__device__ void RK4_classical(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	double u[8];  // velocity placeholders
	double k1[2], k2[2], k3[2], k4[2];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 4);

	// k1 = u_tilde(X,t_n+1) = 4*u_n - 6*u_n-1 + 4*u_n-2 - 1*u_n-3
	k1[0] = 4 * u[0] + -6 * u[2] + 4 * u[4] + -1 * u[6];
	k1[1] = 4 * u[1] + -6 * u[3] + 4 * u[5] + -1 * u[7];

	// compute u_tilde(x - dt*k1/2, t_n+1/2)
	k2[0] = x_in[0] - dt*k1[0]/2.0; k2[1] = x_in[1] - dt*k1[1]/2.0;
	device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 4);

	//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
	k2[0] = 2.1875 * u[0] + -2.1875 * u[2] + 1.3125 * u[4] + -0.3125 * u[6];
	k2[1] = 2.1875 * u[1] + -2.1875 * u[3] + 1.3125 * u[5] + -0.3125 * u[7];

	// compute u_tilde(x - dt*k1/2, t_n+1/2)
	k3[0] = x_in[0] - dt*k2[0]/2.0; k3[1] = x_in[1] - dt*k2[1]/2.0;
	device_hermite_interpolate_grad_2D(phi, k3, u, NX, NY, h, 4);

	//k3 = u_tilde(x - k2 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
	k3[0] = 2.1875 * u[0] + -2.1875 * u[2] + 1.3125 * u[4] + -0.3125 * u[6];
	k3[1] = 2.1875 * u[1] + -2.1875 * u[3] + 1.3125 * u[5] + -0.3125 * u[7];

	//compute u_tilde(x - dt*k3, t_n)
	k4[0] = x_in[0] - dt*k3[0]; k4[1] = x_in[1] - dt*k3[1];
	device_hermite_interpolate_grad_2D(phi, k4, u, NX, NY, h, 1);

	// k4 = u_tilde(x - k3 dt, t_n) = u

	// build all RK-steps together
	x_out[0] = x_in[0] - dt * (k1[0] + 2*k2[0] + 2*k3[0] + u[0])/6.0;
	x_out[1] = x_in[1] - dt * (k1[1] + 2*k2[1] + 2*k3[1] + u[1])/6.0;
}


/*
 * Modified RKThree with negative times, faster than classical
 * Butcher tableau:
 * 1 |  1
 * 2 |  4   -2
 *   | 5/12 2/3 -1/12
 */
__device__ void RK3_optimized(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	long int N = NX * NY;
	double u[6];  // velocity placeholders
	double k1[2], k2[2], k3[2];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 3);

	// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
	k1[0] = 3 * u[0] + -3 * u[2] + u[4];
	k1[1] = 3 * u[1] + -3 * u[3] + u[5];

	// compute u_tilde(X - dt*k1, t_n)
	k2[0] = x_in[0] - dt*k1[0]; k2[1] = x_in[1] - dt*k1[1];
	device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 1);
	// k2 = u_tilde(X - dt*k1, t_n) = u_n

	// compute u_tilde(X - 4*dt*k1 + 2*dt*k2, t_n-1)
	k3[0] = x_in[0] + 2*dt*(-2*k1[0] + u[0]); k3[1] = x_in[1] + 2*dt*(-2*k1[1] + u[1]);
	device_hermite_interpolate_grad_2D(phi + 4*N, k3, u + 2, NX, NY, h, 1);
	// k3 = u_tilde(X - 4*dt*k1 + 2*dt*k2, t_n-1) = u_n-1

	// build all RK-steps together
	// x = x_n - dt (5*k1 + 8*k2 - k3)/12
	x_out[0] = x_in[0] - dt * (5*k1[0] + 8*u[0] - u[2])/12.0;
	x_out[1] = x_in[1] - dt * (5*k1[1] + 8*u[1] - u[3])/12.0;
}


/*
 *  modified RKFour time step of case IV with b=1/4 (Butcher (2016)), is faster than classical
 *  Butcher tableau:
 *   1 |   1
 *  1/2 | 3/8  1/8
 *   1  |  0  -1/3  4/3
 *      | 1/6 -1/12 2/3 1/4
 */
__device__ void RK4_optimized(double *phi, double *x_in, double *x_out, double NX, double NY, double h, double dt) {
	double u[8];  // velocity placeholders
	double k1[2], k2[2], k3[2], k4[2];  // step placeholders

	// compute u_tilde(X,t_n+1)
	device_hermite_interpolate_grad_2D(phi, x_in, u, NX, NY, h, 4);

	// k1 = u_tilde(X,t_n+1) = 4*u_n - 6*u_n-1 + 4*u_n-2 - 1*u_n-3
	k1[0] = 4 * u[0] + -6 * u[2] + 4 * u[4] + -1 * u[6];
	k1[1] = 4 * u[1] + -6 * u[3] + 4 * u[5] + -1 * u[7];

	// compute u_tilde(X - dt*k1, t_n)
	k2[0] = x_in[0] - dt*k1[0]; k2[1] = x_in[1] - dt*k1[1];
	device_hermite_interpolate_grad_2D(phi, k2, u, NX, NY, h, 1);
	// k2 = u_tilde(X - dt*k1, t_n) = u_n
	k2[0] = u[0];
	k2[1] = u[1];

	// compute u_tilde(X - 3/8 dt*k1 - 1/8 dt*k2, t_n+1/2)
	k3[0] = x_in[0] - dt*(3*k1[0] + k2[0])/8.0; k3[1] = x_in[1] - dt*(3*k1[1] + k2[1])/8.0;
	device_hermite_interpolate_grad_2D(phi, k3, u, NX, NY, h, 4);

	//k3 = u_tilde(x - k2 dt/2, t_n+1 - dt/2) = 2.1875*u_n - 2.1875*u_n-1 + 1.3125*u_n-2 - 0.3125*u_n-3
	k3[0] = 2.1875 * u[0] + -2.1875 * u[2] + 1.3125 * u[4] + -0.3125 * u[6];
	k3[1] = 2.1875 * u[1] + -2.1875 * u[3] + 1.3125 * u[5] + -0.3125 * u[7];

	//compute u_tilde(X + 1/3 dt*k2 - 4/3 dt*k3, t_n)
	k4[0] = x_in[0] + dt*(k2[0] - 4*k3[0])/3.0; k4[1] = x_in[1] + dt*(k2[1] - 4*k3[1])/3.0;
	device_hermite_interpolate_grad_2D(phi, k4, u, NX, NY, h, 1);

	// k3 = u_tilde(X + 1/3 dt*k2 - 4/3 dt*k3, t_n) = u

	// build all RK-steps together
	// x = x_n - dt (2*k1 - k2 + 8*k3 + 3*k4)/12
	x_out[0] = x_in[0] - dt * (2*k1[0] - k2[0] + 8*k3[0] + 3*u[0])/12.0;
	x_out[1] = x_in[1] - dt * (2*k1[1] - k2[1] + 8*k3[1] + 3*u[1])/12.0;
}
