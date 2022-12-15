/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
******************************************************************************************************************************/

#include "cmm-simulation-kernel.h"

#include "stdio.h"

#include "../numerical/cmm-hermite.h"
#include "../numerical/cmm-timestep.h"
#include "../simulation/cmm-init.h"


__constant__ double d_L1[4], d_L12[4], d_c1[12], d_cx[12], d_cy[12], d_cxy[12], d_bounds[4];


// init uniform map
__global__ void k_init_diffeo(double *ChiX, double *ChiY, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = iY*Grid.NX + iX;

	ChiX[In] = iX*Grid.hx; ChiY[In] = iY*Grid.hy;

	// x dx = y dy = 1
	ChiX[1*Grid.N+In] = ChiY[2*Grid.N+In] = 1;

	// x dy = y dx = dxdy = 0
	ChiX[2*Grid.N+In] = ChiY[1*Grid.N+In] = ChiX[3*Grid.N+In] = ChiY[3*Grid.N+In] = 0;
}
// double2 variant
__global__ void k_init_diffeo(double2 *Chi, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = iY*Grid.NX + iX;

	Chi[In].x = iX*Grid.hx; Chi[In].y = iY*Grid.hy;

	// x dx = y dy = 1
	Chi[1*Grid.N+In].x = Chi[2*Grid.N+In].y = 1;

	// x dy = y dx = dxdy = 0
	Chi[2*Grid.N+In].x = Chi[1*Grid.N+In].y = Chi[3*Grid.N+In].x = Chi[3*Grid.N+In].y = 0;
}


// swap two memory positions, neede for changing direction in initialization (just simplifies things)
// this is only for hermite functions!
__global__ void k_swap_h(double *Val1, double *Val2, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = 4 * (iY*Grid.NX + iX);  // 4-times as we apply on hermites

	double temp[4] = {Val1[In  ], Val1[In+1], Val1[In+2], Val1[In+3]};
	Val1[In  ] = Val2[In  ]; Val1[In+1] = Val2[In+1]; Val1[In+2] = Val2[In+2]; Val1[In+3] = Val2[In+3];
	Val2[In  ] = temp[0];    Val2[In+1] = temp[1];    Val2[In+2] = temp[2];    Val2[In+3] = temp[3];

}


// sample from hermite of map on uniform grid
__global__ void k_h_sample_map(double *ChiX, double *ChiY, double *ChiX_s, double *ChiY_s, TCudaGrid2D Grid_map, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = iY*Grid.NX + iX;

	//position
	double x = Grid.bounds[0] + iX*Grid.hx;
	double y = Grid.bounds[2] + iY*Grid.hy;

	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);

	ChiX_s[In] = x;
	ChiY_s[In] = y;
}

// sample from set of input points
__global__ void k_h_sample_points_map(TCudaGrid2D Grid_map, TCudaGrid2D Grid, double *ChiX, double *ChiY, double* Dev_particles_pos_in, double* Dev_particles_pos_out, int particle_num)
{
	//index
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position
	// return if position is larger than particle size
	if (i >= particle_num) return;

	double in_x = Dev_particles_pos_in[2*i], in_y = Dev_particles_pos_in[2*i+1];
	double LX = Grid_map.bounds[1] - Grid_map.bounds[0]; double LY = Grid_map.bounds[3] - Grid_map.bounds[2];
	double out_x, out_y;
	// sample from hermite if position is inside map bounds
	if (in_x > Grid.bounds[0] and in_x < Grid.bounds[1] and in_y > Grid.bounds[2] and in_y < Grid.bounds[3]) {
		device_diffeo_interpolate_2D(ChiX, ChiY, in_x, in_y, &out_x, &out_y, Grid_map);

		Dev_particles_pos_out[2*i  ] = out_x - floor(out_x/LX)*LX;
		Dev_particles_pos_out[2*i+1] = out_y - floor(out_y/LY)*LY;
	}
	else  // in case not in bounds, we cannot translate the maps and set the values to NaN
	{
		Dev_particles_pos_out[2*i  ] = Dev_particles_pos_out[2*i+1] = 0.0/0.0;
	}
}


// sample from hermite
__global__ void k_h_sample(double *val, double *val_s, TCudaGrid2D Grid, TCudaGrid2D Grid_s)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid_s.NX || iY >= Grid_s.NY) return;
	int In = iY*Grid_s.NX + iX;

	//position
	double x = Grid_s.bounds[0] + iX*Grid_s.hx;
	double y = Grid_s.bounds[2] + iY*Grid_s.hy;

	val_s[In] = device_hermite_interpolate_2D(val, x, y, Grid);
}

 // sample from hermite given specific input points
__global__ void k_h_sample_points_dxdy(TCudaGrid2D Grid_map, TCudaGrid2D Grid, double *val_h, double* Dev_particles_pos_in, double* Dev_particles_pos_out, int particle_num)
{
	//index
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position
	// return if position is larger than particle size
	if (i >= particle_num) return;

	double in_x = Dev_particles_pos_in[2*i], in_y = Dev_particles_pos_in[2*i+1];
	// sample from hermite if position is inside map bounds
	if (in_x > Grid.bounds[0] and in_x < Grid.bounds[1] and in_y > Grid.bounds[2] and in_y < Grid.bounds[3]) {
		Dev_particles_pos_out[2*i  ] = device_hermite_interpolate_dx_2D(val_h, in_x, in_y, Grid_map);
		Dev_particles_pos_out[2*i+1] = device_hermite_interpolate_dy_2D(val_h, in_x, in_y, Grid_map);
	}
	else  // in case not in bounds, we cannot translate the maps and set the values to NaN
	{
		Dev_particles_pos_out[2*i  ] = Dev_particles_pos_out[2*i+1] = 0.0/0.0;
	}
}


// incompressibility check by computing determinant of jacobi of map
__global__ void k_incompressibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_map,
		double *ChiX, double *ChiY, double *grad_Chi)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid_check.NX || iY >= Grid_check.NY) return;
	int In = iY*Grid_check.NX + iX;

	//position shifted by half a point to compute at off-grid, i'm not tooo sure if its important though
	double x = iX*Grid_check.hx + 0.5*Grid_check.hx;
	double y = iY*Grid_check.hy + 0.5*Grid_check.hy;
//	double x = iX*Grid_check.hx;
//	double y = iY*Grid_check.hy;
	grad_Chi[In] = device_diffeo_grad_2D(ChiX, ChiY, x, y, Grid_map);
}


// invertibility check by computing distance from forth-and-back mapping to original point, starting with forward to backward map
__global__ void k_invertibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_backward, TCudaGrid2D Grid_forward,
		double *ChiX_b, double *ChiY_b, double *ChiX_f, double *ChiY_f, double *abs_invert)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid_check.NX || iY >= Grid_check.NY) return;
	int In = iY*Grid_check.NX + iX;

	//position shifted by half a point to compute at off-grid, i'm not tooo sure if its important though
	double x = iX*Grid_check.hx + 0.5*Grid_check.hx;
	double y = iY*Grid_check.hy + 0.5*Grid_check.hy;
//	double x = iX*Grid_check.hx;
//	double y = iY*Grid_check.hy;

	double x_inv, y_inv;  // initialize mapped points
	device_diffeo_interpolate_2D(ChiX_f, ChiY_f, x, y, &x_inv, &y_inv, Grid_forward);
	device_diffeo_interpolate_2D(ChiX_b, ChiY_b, x_inv, y_inv, &x_inv, &y_inv, Grid_backward);

	// I am actually not yet sure how to compute the error, but I believe that describing it by the euclidean distance could be the best way
	abs_invert[In] = hypot(x-x_inv, y-y_inv);
}


/*
 * Main advection function of the flow map using the stream function
 * Loop over footpoints to apply GALS
 * For each foot point: advect using the stream function and time stepping scheme
 * At the end: combine results of all footpoints using specific map update scheme
 */
__global__ void k_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi,
		TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num, int l_order, int direction)
{
//	if ((blockIdx.x == 0) && (blockIdx.y == 0)) { debug = true; }

	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
    if(iX >= Grid_map.NX || iY >= Grid_map.NY) return;
    int In = iY*Grid_map.NX + iX;

	//running through neighbours
	double x_ep[2], x_f[2];

	// initialize new intermediate values as zeros, helpful to not write to array every point
	double Chi_full_x[4] = {0, 0, 0, 0};
	double Chi_full_y[4] = {0, 0, 0, 0};

	// repeat for all footpoints, 4 for 2th order, 8 for 4th order and 12 for 6th order
	for (int k_foot = 0; k_foot< map_update_order_num*4 + 4; k_foot++) {
		// get position of footpoint, NE, SE, SW, NW
		// for higher orders repeat cross shape stencil with more points, bitwise operators for integer /2 and %2
		x_ep[0] = Grid_map.bounds[0] + iX*Grid_map.hx + ((k_foot>>2) + 1) * ep*(1 - 2*((k_foot>>1)&1));
		x_ep[1] = Grid_map.bounds[2] + iY*Grid_map.hy + ((k_foot>>2) + 1) * ep*(1 - 2*(((k_foot+1)>>1)&1));

		if (direction == -1)  // time integration backwards in time for backwards map
		{
			switch (time_integration_num) {
				case 10: { euler_exp_b         (psi, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// ABTwo
				case 20: { adam_bashford_2_pc_b(psi, x_ep, x_f, Grid_psi, dt         ); break; }
				// ABTwo
				case 21: { RK2_heun_b          (psi, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// RKThree
				case 30: { RK3_classical_b     (psi, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// RKFour
				case 40: { RK4_classical_b     (psi, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// custom RKThree
				case 31: { RK3_optimized_b     (psi, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// custom RKFour case IV
				case 41: { RK4_optimized_b     (psi, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// zero on default
				default: { x_f[0] = x_f[1] = 0; }
			}
			// apply map deformation to go back to inital point
			device_diffeo_interpolate_2D(ChiX, ChiY, x_f[0], x_f[1], &x_f[0], &x_f[1], Grid_map);
		}
		else if (direction == 1)  // time integration forwards in time for forwards map
		{
			// apply map deformation to go to current time step
			device_diffeo_interpolate_2D(ChiX, ChiY, x_ep[0], x_ep[1], &x_f[0], &x_f[1], Grid_map);

			switch (time_integration_num) {
				case 10: { euler_exp         (psi, x_f, x_f, Grid_psi, dt         ); break; }
				// ABTwo
				case 20: { adam_bashford_2_pc(psi, x_f, x_f, Grid_psi, dt         ); break; }
				// ABTwo
				case 21: { RK2_heun          (psi, x_f, x_f, Grid_psi, dt, l_order); break; }
				// RKThree
				case 30: { RK3_classical     (psi, x_f, x_f, Grid_psi, dt, l_order); break; }
				// RKFour
				case 40: { RK4_classical     (psi, x_f, x_f, Grid_psi, dt, l_order); break; }
				// custom RKThree
				case 31: { RK3_optimized     (psi, x_f, x_f, Grid_psi, dt         ); break; }
				// custom RKFour case IV
				case 41: { RK4_optimized     (psi, x_f, x_f, Grid_psi, dt, l_order); break; }
				// zero on default
				default: { x_f[0] = x_f[1] = 0; }
			}
		}

		// directly apply map update
		// chi values - central average with stencil +NE +SE +SW +NW
		Chi_full_x[0] += x_f[0] * d_c1 [k_foot]; Chi_full_y[0] += x_f[1] * d_c1 [k_foot];

		// chi grad x - central differences with stencil +NE +SE -SW -NW
		Chi_full_x[1] += x_f[0] * d_cx [k_foot]; Chi_full_y[1] += x_f[1] * d_cx [k_foot];

		// chi grad y - central differences with stencil +NE -SE -SW +NW
		Chi_full_x[2] += x_f[0] * d_cy [k_foot]; Chi_full_y[2] += x_f[1] * d_cy [k_foot];

		// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		Chi_full_x[3] += x_f[0] * d_cxy[k_foot]; Chi_full_y[3] += x_f[1] * d_cxy[k_foot];

	}

	// can't use Chi because we still use it for diffeo_interpolate
	Chi_new_X[             In] = Chi_full_x[0];	Chi_new_Y[             In] = Chi_full_y[0];
	Chi_new_X[1*Grid_map.N+In] = Chi_full_x[1];	Chi_new_Y[1*Grid_map.N+In] = Chi_full_y[1];
	Chi_new_X[2*Grid_map.N+In] = Chi_full_x[2];	Chi_new_Y[2*Grid_map.N+In] = Chi_full_y[2];
	Chi_new_X[3*Grid_map.N+In] = Chi_full_x[3];	Chi_new_Y[3*Grid_map.N+In] = Chi_full_y[3];
}


// apply first map
// like k_h_sample_map but saves all results in one array next to each other
__global__ void k_h_sample_map_compact(double *ChiX, double *ChiY, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = iY*Grid.NX + iX;

	//position
	double x = Grid.bounds[0] + iX*Grid.hx;
	double y = Grid.bounds[2] + iY*Grid.hy;
	double LX = Grid_map.bounds[1] - Grid_map.bounds[0]; double LY = Grid_map.bounds[3] - Grid_map.bounds[2];

	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);

	// save in two points in array
	x_y[2*In  ] = x - floor(x / LX) * LX;
	x_y[2*In+1] = y - floor(y / LY) * LY;
}
// apply intermediate maps to compacted map, basically only samples in stacked form
__global__ void k_apply_map_compact(double *ChiX_stack, double *ChiY_stack, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = iY*Grid.NX + iX;

	double points[2];
	double LX = Grid_map.bounds[1] - Grid_map.bounds[0]; double LY = Grid_map.bounds[3] - Grid_map.bounds[2];
	device_diffeo_interpolate_2D(ChiX_stack, ChiY_stack, x_y[2*In], x_y[2*In+1], &points[0], &points[1], Grid_map);
	x_y[2*In] = points[0] - floor(points[0]/LX)*LX;
	x_y[2*In+1] = points[1] - floor(points[1]/LY)*LY;
}
// sample from initial condition
__global__ void k_h_sample_from_init(double *Val_out, double *x_y, TCudaGrid2D Grid, TCudaGrid2D Grid_discrete, int init_var_num, int init_num, double *Val_initial_discrete, bool initial_discrete)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid.NX || iY >= Grid.NY) return;
	int In = iY*Grid.NX + iX;

	if (!initial_discrete) {
		// differentiate between what variable we would like to retrieve
		switch (init_var_num) {
			case 0: { Val_out[In] = d_init_vorticity(x_y[2*In], x_y[2*In+1], init_num); break; }
			case 1: { Val_out[In] = d_init_scalar(x_y[2*In], x_y[2*In+1], init_num); break; }
		}
	}
	else {
		Val_out[In] = device_hermite_interpolate_2D(Val_initial_discrete, x_y[2*In], x_y[2*In+1], Grid_discrete);
	}
}


// this function is nice, but not implemented for cpu mapstack yet
__global__ void k_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, TCudaGrid2D Grid_map, TCudaGrid2D Grid_fine, int simulation_num)
{
//	//index
//	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
//	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
//
//	if(iX >= Grid_fine.NX || iY >= Grid_fine.NY)
//		return;
//
//	int In = iY*Grid_fine.NX + iX;
//
//	//position
//	double x = iX*Grid_fine.hx;
//	double y = iY*Grid_fine.hy;
//
//	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);
//	for(int k = stack_length - 1; k >= 0; k--)
//		device_diffeo_interpolate_2D(&ChiX_stack[k*Grid_map.N*4], &ChiY_stack[k*Grid_map.N*4], x, y, &x, &y, Grid_map);
//
//	error[In] = fabs(d_init_vorticity(x, y, simulation_num) - d_init_vorticity(iX*Grid_fine.hx, iY*Grid_fine.hy, simulation_num));
}



// apply map, sample from fine vorticity and apply mollifier, this function is basically only used by evaluate_stream
__global__ void k_apply_map_and_sample_from_hermite(double *ChiX, double *ChiY, double *fs, double *H, TCudaGrid2D Grid_map, TCudaGrid2D Grid_vort, TCudaGrid2D Grid_fine, int molly_stencil, bool padd_inline_fft)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	if(iX >= Grid_vort.NX || iY >= Grid_vort.NY) return;

	int In;
	// for evaluate stream hermite we need a specific padding for inline fft
	if (!padd_inline_fft) In = iY*Grid_vort.NX + iX;
	else In = iY*Grid_vort.NX_fft*2 + iX;

	//position
	double x = iX*Grid_vort.hx; double y = iY*Grid_vort.hy;

	// mollification to act as a lowpass filter
	double x2, y2;

	/*                                      0  1/6  0
	 * mollifier of order 1 using stencil  1/6 1/3 1/6
	 * using 4 neighbouring points			0  1/6  0
	 */
	if (molly_stencil == 4) {
		// compute main points
		device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x2, &y2, Grid_map);
		double moll_add = device_hermite_interpolate_2D(H, x2, y2, Grid_fine)/3.0;  // other values will be added on here
		for (int i_molly = 0; i_molly < 4; i_molly++) {
			// choose 4 points in between the grid: W, E, S, N, &1 and >>1 for %2 and //2
			x2 = x + Grid_vort.hx/2.0*((i_molly>>1+1)&1) * (-1 + 2*(i_molly&1));  // -1 +1  0  0
			y2 = y + Grid_vort.hy/2.0*((i_molly>>1  )&1) * (-1 + 2*(i_molly&1));  //  0  0 -1 +1

			device_diffeo_interpolate_2D(ChiX, ChiY, x2, y2, &x2, &y2, Grid_map);
			moll_add += device_hermite_interpolate_2D(H, x2, y2, Grid_fine)/6.0;
		}
		fs[In] = moll_add;
	}
	/*                                     1/16 2/16 1/16
	 * mollifier of order 1 using stencil  2/16 4/16 2/16
	 * using 8 neighbouring points		   1/16 2/16 1/16
	 */
	else if (molly_stencil == 8) {
//		double moll_fac[9] = {1/16, 2/16, 1/16, 2/16, 4/16, 2/16, 1/16, 2/16, 1/16};  // molly factor for all points, chosen randomly
		// compute main points
		double moll_add = 0;  // other values will be added on here
		for (int i_molly = 0; i_molly < 9; i_molly++) {
			// choose 9 points in between the grid: SW, S, SE, W, C, E, NW, N, NE
			x2 = x + Grid_vort.hx*(-1 + i_molly%3)/2.0;
			y2 = y + Grid_vort.hy*(-1 + i_molly/3)/2.0;

			device_diffeo_interpolate_2D(ChiX, ChiY, x2, y2, &x2, &y2, Grid_map);
			moll_add += (1 + (i_molly%3)&1) * (1 + (i_molly/3)&1) * device_hermite_interpolate_2D(H, x2, y2, Grid_fine)/16.0;
		}
		fs[In] = moll_add;
	}
	// else, assume no mollification
	else {
		device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);
		fs[In] = device_hermite_interpolate_2D(H, x, y, Grid_fine);
	}
}
