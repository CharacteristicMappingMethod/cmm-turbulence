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

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX + iX;

	ChiX[In] = iX*Grid.hx;
	ChiY[In] = iY*Grid.hy;

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

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX + iX;

	Chi[In].x = iX*Grid.hx;
	Chi[In].y = iY*Grid.hy;

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

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

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

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX + iX;

	//position
	double x = Grid.bounds[0] + iX*Grid.hx;
	double y = Grid.bounds[2] + iY*Grid.hy;

	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);

	ChiX_s[In] = x;
	ChiY_s[In] = y;
}

// sample from set of input points
__global__ void k_h_sample_points_map(TCudaGrid2D Grid_map, double *ChiX, double *ChiY, double* Dev_particles_pos_in, double* Dev_particles_pos_out, int particle_num)
{
	//index
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	double in_x = Dev_particles_pos_in[2*i], in_y = Dev_particles_pos_in[2*i+1];
	double out_x, out_y;
	// sample from hermite if position is inside map bounds
	if (in_x > Grid_map.bounds[0] and in_x < Grid_map.bounds[1] and in_y > Grid_map.bounds[2] and in_y < Grid_map.bounds[3]) {
		device_diffeo_interpolate_2D(ChiX, ChiY, in_x, in_y, &out_x, &out_y, Grid_map);

		Dev_particles_pos_out[2*i  ] = out_x;
		Dev_particles_pos_out[2*i+1] = out_y;
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

	if(iX >= Grid_s.NX || iY >= Grid_s.NY)
		return;

	int In = iY*Grid_s.NX + iX;

	//position
	double x = Grid_s.bounds[0] + iX*Grid_s.hx;
	double y = Grid_s.bounds[2] + iY*Grid_s.hy;

	val_s[In] = device_hermite_interpolate_2D(val, x, y, Grid);
}


// incompressibility check by computing determinant of jacobi of map
__global__ void k_incompressibility_check(TCudaGrid2D Grid_check, TCudaGrid2D Grid_map,
		double *ChiX, double *ChiY, double *grad_Chi)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid_check.NX || iY >= Grid_check.NY)
		return;

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

	if(iX >= Grid_check.NX || iY >= Grid_check.NY)
		return;

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


// compute footpoints at exact grid locations
__global__ void k_compute_footpoints(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi,
		TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, int time_integration_num, int l_order, int direction) {
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
    /*int NX = 512; //Dans l'interpolation remettre NXc Nyc hc
    int NY = 512;
    double h = twoPI/(float)NX;*/

    if(iX >= Grid_map.NX || iY >= Grid_map.NY)
		return;

    int In = iY*Grid_map.NX + iX;

    double x_ep[2], x_f[2];

	x_ep[0] = iX*Grid_map.hx; x_ep[1] = iY*Grid_map.hy;

	if (direction == -1)  // time integration backwards in time for backwards map
	{
		switch (time_integration_num) {
			case 10: { euler_exp_b         (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// ABTwo
			case 20: { adam_bashford_2_pc_b(psi,              x_ep, x_f, Grid_psi, dt         ); break; }
			// ABTwo
			case 21: { RK2_heun_b          (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// RKThree
			case 30: { RK3_classical_b     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// RKFour
			case 40: { RK4_classical_b     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// custom RKThree
			case 31: { RK3_optimized_b     (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// custom RKFour case IV
			case 41: { RK4_optimized_b     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// zero on default
			default: { x_f[0] = x_f[1] = 0; }
		}
		// apply map deformation to go back to inital point
		device_diffeo_interpolate_2D(ChiX, ChiY, x_f[0], x_f[1], &x_f[0], &x_f[1], Grid_map);
	}
	else if (direction == 1)  // time integration forwards in time for forwards map
	{
		// apply map deformation to go to current time step
		device_diffeo_interpolate_2D(ChiX, ChiY, x_f[0], x_f[1], &x_f[0], &x_f[1], Grid_map);

		switch (time_integration_num) {
			case 10: { euler_exp         (psi,              x_ep, x_f, Grid_psi, dt         ); break; }
			// ABTwo
			case 20: { adam_bashford_2_pc(psi,              x_ep, x_f, Grid_psi, dt         ); break; }
			// ABTwo
			case 21: { RK2_heun          (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// RKThree
			case 30: { RK3_classical     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// RKFour
			case 40: { RK4_classical     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// custom RKThree
			case 31: { RK3_optimized     (psi,              x_ep, x_f, Grid_psi, dt         ); break; }
			// custom RKFour case IV
			case 41: { RK4_optimized     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// zero on default
			default: { x_f[0] = x_f[1] = 0; }
		}
	}

	// transcribe
	Chi_new_X[In] = x_f[0];	Chi_new_Y[In] = x_f[1];
}


// update map by grid itself, this should hopefully use some funky cuda techniques
// it works, but is not accurate enough in comparison to footpoint method
// do template for map update orders to help the compiler
__global__ void k_map_update(double *Chi, double *Chi_new, TCudaGrid2D Grid_map, int map_ord_1, int warp_direc) {
	// shared memory for loading the data - block widened by map_ord_1
    extern __shared__ double sdata[];

	int thread_id_xy = threadIdx.x + threadIdx.y * blockDim.x;  // unique ID of thread in block

	// get positions in whole grid
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	bool debug = false;
//	if ((blockIdx.x == 0) && (blockIdx.y == 0)) { debug = true; }

    if(iX >= Grid_map.NX || iY >= Grid_map.NY) return;  // safety precaution

    // index in arrays
    int In = iY*Grid_map.NX + iX;
    long int N = Grid_map.NX*Grid_map.NY;

	if (debug) {
		printf("Id : %d \t block x : %d \t block y : %d \t In : %d\n", thread_id_xy, iX, iY, In);
	}

    int ring_num = (16 + 2 * map_ord_1)* (16 + 2* map_ord_1) - 256;  // how many elements do we have in the ring
    int change_num = (16 + 2 * map_ord_1) * map_ord_1;  // how many elements at top or bottom

    // load values in shared memory, this makes it efficient as we have to access less storage in total
    // central block first, each thread loads memory at its location, offset to fit
    int sdata_pos = change_num + map_ord_1 + thread_id_xy + 2*map_ord_1*(thread_id_xy / 16);
    sdata[sdata_pos] = Chi_new[In];

	if (debug) {
		printf("Id : %d \t store 1 : %d \t load 1 : %d\n", thread_id_xy, sdata_pos, In);
		__syncthreads();  // only for debugging
	}

	if (thread_id_xy < ring_num) {
		// load in ring values computed without ifs so that all important threads have something to do
		int load_id, store_id;

		// where should we store in shared memory?
		store_id = thread_id_xy
				 + ((thread_id_xy >= change_num)*(thread_id_xy < ring_num-change_num) * ((thread_id_xy + map_ord_1 - change_num)/(2*map_ord_1))*16)
				 + (thread_id_xy >= ring_num-change_num)*256;

		// due to periodic boundaries, we have to check for x and y values to perform waprs
		int load_differ_x = - map_ord_1 + (thread_id_xy < change_num)*(thread_id_xy%(16+map_ord_1*2))
				+ (thread_id_xy < ring_num-change_num)*(thread_id_xy >= change_num) * (thread_id_xy%(2*map_ord_1) + ((thread_id_xy/map_ord_1)%2)*16)
				+ (thread_id_xy >= ring_num-change_num)*((thread_id_xy-ring_num+change_num)%(16+map_ord_1*2));
		int load_differ_y = - map_ord_1 + (thread_id_xy < change_num)*thread_id_xy/(16+map_ord_1*2)
				+ (thread_id_xy < ring_num-change_num)*(thread_id_xy >= change_num) * (map_ord_1+(thread_id_xy - change_num)/(2*map_ord_1))
				+ (thread_id_xy >= ring_num-change_num)*(16+map_ord_1+(thread_id_xy-ring_num+change_num)/(16+map_ord_1*2));

		if (debug) {
			printf("Id : %d \t load_x : %d \t load y : %d\n", thread_id_xy, load_differ_x, load_differ_y);
		}

		// compute
		int load_id_x = blockDim.x * blockIdx.x + load_differ_x;
		int load_id_y = blockDim.y * blockIdx.y + load_differ_y;

		int warp_x = (load_id_x/Grid_map.NX - (load_id_x < 0));  // needed for warping the values back
		int warp_y = (load_id_y/Grid_map.NY - (load_id_y < 0));  // needed for warping the values back
		load_id_x -= warp_x*Grid_map.NX;  // warp
		load_id_y -= warp_y*Grid_map.NY;  // warp

		load_id = load_id_y * Grid_map.NX + load_id_x;

		sdata[store_id] = Chi_new[load_id] + twoPI*((1 - warp_direc)*warp_x + warp_direc*warp_y);

		if (debug) {
			printf("Id : %d \t store 2 : %d \t load 2 : %d\n", thread_id_xy, store_id, load_id);
		}
	}


    // synch threads needed, as different warps compute different things
    __syncthreads();

    // now, lets modify the values
    double chi_new[4];

	// chi values - exact center point
    int shift_y = change_num/map_ord_1;  // construct stencils easily
    chi_new[0] = sdata[sdata_pos];

    switch (map_ord_1) {
    	// second order
		case 1: {
			// chi grad x - central differences with stencil +NE +SE -SW -NW
			chi_new[1] = (sdata[sdata_pos + 1] 			 - sdata[sdata_pos - 1])
					   / (2.0 * Grid_map.h);

			// chi grad y - central differences with stencil +NE -SE -SW +NW
		    chi_new[2] = (sdata[sdata_pos + shift_y] 	 - sdata[sdata_pos - shift_y])
		    		   / (2.0 * Grid_map.h);

			// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		    chi_new[3] = (sdata[sdata_pos + shift_y + 1] - sdata[sdata_pos - shift_y + 1] + sdata[sdata_pos - shift_y - 1]  - sdata[sdata_pos + shift_y - 1])
		    		   / (4.0 * Grid_map.h * Grid_map.h);
			break;
		}
		// fourth order
		case 2: {
			// chi grad x - central differences with stencil +NE +SE -SW -NW
			chi_new[1] = (8*(sdata[sdata_pos + 1] 			   - sdata[sdata_pos - 1])
					   -    (sdata[sdata_pos + 2] 			   - sdata[sdata_pos - 2]))
					   / (12.0 * Grid_map.h);

			// chi grad y - central differences with stencil +NE -SE -SW +NW
		    chi_new[2] = (8*(sdata[sdata_pos +   shift_y] 	   - sdata[sdata_pos -   shift_y])
		    		   -    (sdata[sdata_pos + 2*shift_y] 	   - sdata[sdata_pos - 2*shift_y]))
					   / (12.0 * Grid_map.h);

			// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		    chi_new[3] =  (8*(sdata[sdata_pos +   shift_y + 1] - sdata[sdata_pos -   shift_y + 1] + sdata[sdata_pos -   shift_y - 1]  - sdata[sdata_pos +   shift_y - 1])
		    		   -     (sdata[sdata_pos + 2*shift_y + 2] - sdata[sdata_pos - 2*shift_y + 2] + sdata[sdata_pos - 2*shift_y - 2]  - sdata[sdata_pos + 2*shift_y - 2]))
		    	       / (24.0 * Grid_map.h * Grid_map.h);
			break;
		}
		// sixth order
		case 3: {
			// chi grad x - central differences with stencil +NE +SE -SW -NW
			chi_new[1] = (45*(sdata[sdata_pos + 1] 				- sdata[sdata_pos - 1])
					   -   9*(sdata[sdata_pos + 2] 				- sdata[sdata_pos - 2])
					   +     (sdata[sdata_pos + 3] 			    - sdata[sdata_pos - 3]))
					   / (60.0 * Grid_map.h);

			// chi grad y - central differences with stencil +NE -SE -SW +NW
		    chi_new[2] = (45*(sdata[sdata_pos +   shift_y] 		- sdata[sdata_pos -   shift_y])
		    		   -   9*(sdata[sdata_pos + 2*shift_y] 		- sdata[sdata_pos - 2*shift_y])
					   +     (sdata[sdata_pos + 3*shift_y] 		- sdata[sdata_pos - 3*shift_y]))
					   / (60.0 * Grid_map.h);

			// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		    chi_new[3] =  (45*(sdata[sdata_pos +   shift_y + 1] - sdata[sdata_pos -   shift_y + 1] + sdata[sdata_pos -   shift_y - 1]  - sdata[sdata_pos +   shift_y - 1])
		    		   -   18*(sdata[sdata_pos + 2*shift_y + 2] - sdata[sdata_pos - 2*shift_y + 2] + sdata[sdata_pos - 2*shift_y - 2]  - sdata[sdata_pos + 2*shift_y - 2])
		    		   +      (sdata[sdata_pos + 3*shift_y + 3] - sdata[sdata_pos - 3*shift_y + 3] + sdata[sdata_pos - 3*shift_y - 3]  - sdata[sdata_pos + 3*shift_y - 3]))
		    	       / (120.0 * Grid_map.h * Grid_map.h);
			break;
		}
	}

    // transcribe final results
	Chi[    In] = chi_new[0];
	Chi[1*N+In] = chi_new[1];
	Chi[2*N+In] = chi_new[2];
	Chi[3*N+In] = chi_new[3];
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

    if(iX >= Grid_map.NX || iY >= Grid_map.NY)
		return;

	int In = iY*Grid_map.NX + iX;

	//running through neighbours
	double x_ep[2], x_f[2];

	// initialize new intermediate values as zeros, helpful to not write to array every point
	double Chi_full_x[4] = {0, 0, 0, 0};
	double Chi_full_y[4] = {0, 0, 0, 0};

	// repeat for all footpoints, 4 for 2th order, 8 for 4th order and 12 for 6th order
	for (int k_foot = 0; k_foot< map_update_order_num*4 + 4; k_foot++) {
		int i_foot_now = (k_foot/4);  // used for getting
		double i_dist_now = i_foot_now+1;

		// get position of footpoint, NE, SE, SW, NW
		// for higher orders repeat cross shape stencil with more points
		x_ep[0] = Grid_map.bounds[0] + iX*Grid_map.hx + i_dist_now * ep*(1 - 2*((k_foot/2)%2));
		x_ep[1] = Grid_map.bounds[2] + iY*Grid_map.hy + i_dist_now * ep*(1 - 2*(((k_foot+1)/2)%2));

		if (direction == -1)  // time integration backwards in time for backwards map
		{
			switch (time_integration_num) {
				case 10: { euler_exp_b         (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
				// ABTwo
				case 20: { adam_bashford_2_pc_b(psi,              x_ep, x_f, Grid_psi, dt         ); break; }
				// ABTwo
				case 21: { RK2_heun_b          (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
				// RKThree
				case 30: { RK3_classical_b     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// RKFour
				case 40: { RK4_classical_b     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
				// custom RKThree
				case 31: { RK3_optimized_b     (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
				// custom RKFour case IV
				case 41: { RK4_optimized_b     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
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
				case 10: { euler_exp         (psi,              x_f, x_f, Grid_psi, dt         ); break; }
				// ABTwo
				case 20: { adam_bashford_2_pc(psi,              x_f, x_f, Grid_psi, dt         ); break; }
				// ABTwo
				case 21: { RK2_heun          (psi, d_L1,        x_f, x_f, Grid_psi, dt, l_order); break; }
				// RKThree
				case 30: { RK3_classical     (psi, d_L1, d_L12, x_f, x_f, Grid_psi, dt, l_order); break; }
				// RKFour
				case 40: { RK4_classical     (psi, d_L1, d_L12, x_f, x_f, Grid_psi, dt, l_order); break; }
				// custom RKThree
				case 31: { RK3_optimized     (psi,              x_f, x_f, Grid_psi, dt         ); break; }
				// custom RKFour case IV
				case 41: { RK4_optimized     (psi, d_L1, d_L12, x_f, x_f, Grid_psi, dt, l_order); break; }
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

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX + iX;

	//position
	double x = Grid.bounds[0] + iX*Grid.hx;
	double y = Grid.bounds[2] + iY*Grid.hy;

	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);

	// save in two points in array
	x_y[2*In  ] = x;
	x_y[2*In+1] = y;

}
// apply intermediate maps to compacted map, basically only samples in stacked form
__global__ void k_apply_map_compact(double *ChiX_stack, double *ChiY_stack, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX + iX;

	//for(int k = stack_length - 1; k >= 0; k--)
	device_diffeo_interpolate_2D(ChiX_stack, ChiY_stack, x_y[2*In], x_y[2*In+1], &x_y[2*In], &x_y[2*In+1], Grid_map);

}
// sample from initial condition
__global__ void k_h_sample_from_init(double *Val_out, double *x_y, TCudaGrid2D Grid, TCudaGrid2D Grid_discrete, int init_var_num, int init_num, double *Val_initial_discrete, bool initial_discrete)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	long int In = iY*Grid.NX + iX;

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

	if(iX >= Grid_vort.NX || iY >= Grid_vort.NY)
		return;

	int In;
	// for evaluate stream hermite we need a specific padding for inline fft
	if (!padd_inline_fft) In = iY*Grid_vort.NX + iX;
	else In = iY*Grid_vort.NX_fft*2 + iX;

	//position
	double x = iX*Grid_vort.hx;
	double y = iY*Grid_vort.hy;

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
			// choose 4 points in between the grid: W, E, S, N
			x2 = x + Grid_vort.hx/2.0*((i_molly/2+1)%2) * (-1 + 2*(i_molly%2));  // -1 +1  0  0
			y2 = y + Grid_vort.hy/2.0*((i_molly/2  )%2) * (-1 + 2*(i_molly%2));  //  0  0 -1 +1

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
			moll_add += (1 + (i_molly%3)%2) * (1 + (i_molly/3)%2) * device_hermite_interpolate_2D(H, x2, y2, Grid_fine)/16.0;
		}
		fs[In] = moll_add;
	}
	// else, assume no mollification
	else {
		device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);
		fs[In] = device_hermite_interpolate_2D(H, x, y, Grid_fine);
	}
}
