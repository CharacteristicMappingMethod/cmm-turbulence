#include "cmm-simulation-kernel.h"

#include "stdio.h"

#include "../numerical/cmm-hermite.h"
#include "../numerical/cmm-timestep.h"

#include <curand.h>
#include <curand_kernel.h>


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


// sample from hermite of map
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
__global__ void k_incompressibility_check(double *ChiX, double *ChiY, double *gradChi, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid_fine.NX || iY >= Grid_fine.NY)
		return;

	int In = iY*Grid_fine.NX + iX;

	//position shifted by half a point to compute at off-grid
	double x = iX*Grid_fine.hx + 0.5*Grid_fine.hx;
	double y = iY*Grid_fine.hy + 0.5*Grid_fine.hy;
//	double x = iX*Grid_fine.hx;
//	double y = iY*Grid_fine.hy;
	gradChi[In] = device_diffeo_grad_2D(ChiX, ChiY, x, y, Grid_coarse);
}


// compute footpoints at exact grid locations
__global__ void k_compute_footpoints(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *psi, TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, int time_integration_num, int l_order) {
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

	// time integration - note, we go backwards in time!
	switch (time_integration_num) {
		case 10: { euler_exp         (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
		// ABTwo
		case 20: { adam_bashford_2_pc(psi,              x_ep, x_f, Grid_psi, dt         ); break; }
		// ABTwo
		case 21: { RK2_heun          (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
		// RKThree
		case 30: { RK3_classical     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
		// RKFour
		case 40: { RK4_classical     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
		// custom RKThree
		case 31: { RK3_optimized     (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
		// custom RKFour case IV
		case 41: { RK4_optimized     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
		// zero on default
		default: { x_f[0] = x_f[1] = 0; }
	}

	// apply map deformation
	device_diffeo_interpolate_2D(ChiX, ChiY, x_f[0], x_f[1], &x_f[0], &x_f[1], Grid_map);

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
		TCudaGrid2D Grid_map, TCudaGrid2D Grid_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num, int l_order)
{
	bool debug = true;
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

		// time integration - note, we go backwards in time!
		switch (time_integration_num) {
			case 10: { euler_exp         (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// ABTwo
			case 20: { adam_bashford_2_pc(psi,              x_ep, x_f, Grid_psi, dt         ); break; }
			// ABTwo
			case 21: { RK2_heun          (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// RKThree
			case 30: { RK3_classical     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// RKFour
			case 40: { RK4_classical     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// custom RKThree
			case 31: { RK3_optimized     (psi, d_L1,        x_ep, x_f, Grid_psi, dt, l_order); break; }
			// custom RKFour case IV
			case 41: { RK4_optimized     (psi, d_L1, d_L12, x_ep, x_f, Grid_psi, dt, l_order); break; }
			// zero on default
			default: { x_f[0] = x_f[1] = 0; }
		}

		// apply map deformation
		device_diffeo_interpolate_2D(ChiX, ChiY, x_f[0], x_f[1], &x_f[0], &x_f[1], Grid_map);

		// directly apply map update
		// chi values - central average with stencil +NE +SE +SW +NW
		Chi_full_x[0] += x_f[0] * d_c1 [k_foot]; Chi_full_y[0] += x_f[1] * d_c1 [k_foot];

		// chi grad x - central differences with stencil +NE +SE -SW -NW
		Chi_full_x[1] += x_f[0] * d_cx [k_foot]; Chi_full_y[1] += x_f[1] * d_cx [k_foot];

		// chi grad y - central differences with stencil +NE -SE -SW +NW
		Chi_full_x[2] += x_f[0] * d_cy [k_foot]; Chi_full_y[2] += x_f[1] * d_cy [k_foot];

		// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		Chi_full_x[3] += x_f[0] * d_cxy[k_foot]; Chi_full_y[3] += x_f[1] * d_cxy[k_foot];

		if (debug and In == 260863) {
			printf("Id : %d \t ifoot : %d \t fpoint x : %.20f \t fpoint y : %.20f \n", In, k_foot, x_f[0], x_f[1]);
//			printf("Fac: Id : %d \t Val : %.20f \t x : %.20f \t y : %.20f \t xy : %.20f \n", In, d_c1 [k_foot], d_cx [k_foot], d_cy [k_foot], d_cxy[k_foot]);

			printf("X: Id : %d \t Val : %.20f \t x : %.20f \t y : %.20f \t xy : %.20f \n", In, Chi_full_x[0], Chi_full_x[1], Chi_full_x[2], Chi_full_x[3]);
			printf("Y: Id : %d \t Val : %.20f \t x : %.20f \t y : %.20f \t xy : %.20f \n", In, Chi_full_y[0], Chi_full_y[1], Chi_full_y[2], Chi_full_y[3]);
		}
	}

	if (debug and In == 260863) {
		printf("X: Id : %d \t Val : %.20f \t x : %.20f \t y : %.20f \t xy : %.20f \n", In, Chi_full_x[0], Chi_full_x[1], Chi_full_x[2], Chi_full_x[3]);
		printf("Y: Id : %d \t Val : %.20f \t x : %.20f \t y : %.20f \t xy : %.20f \n", In, Chi_full_y[0], Chi_full_y[1], Chi_full_y[2], Chi_full_y[3]);
	}

	// can't use Chi because we still use it for diffeo_interpolate
	Chi_new_X[             In] = Chi_full_x[0];	Chi_new_Y[             In] = Chi_full_y[0];
	Chi_new_X[1*Grid_map.N+In] = Chi_full_x[1];	Chi_new_Y[1*Grid_map.N+In] = Chi_full_y[1];
	Chi_new_X[2*Grid_map.N+In] = Chi_full_x[2];	Chi_new_Y[2*Grid_map.N+In] = Chi_full_y[2];
	Chi_new_X[3*Grid_map.N+In] = Chi_full_x[3];	Chi_new_Y[3*Grid_map.N+In] = Chi_full_y[3];
}


// apply first map
// like k_h_sample_map but saves all results in one array next to each other
__global__ void k_apply_map_stack_to_W_custom_part_1(double *ChiX, double *ChiY, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	int In = iY*Grid.NX + iX;

	double h_temp_x = (Grid.bounds[1] - Grid.bounds[0])/Grid.NX;
	double h_temp_y = (Grid.bounds[3] - Grid.bounds[2])/Grid.NY;

	//position
	double x = Grid.bounds[0] + iX * h_temp_x;
	double y = Grid.bounds[2] + iY * h_temp_y;

	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);

	// save in two points in array
	x_y[2*In  ] = x;
	x_y[2*In+1] = y;

}
// apply intermediate maps
__global__ void k_apply_map_stack_to_W_part_2(double *ChiX_stack, double *ChiY_stack, double *x_y, TCudaGrid2D Grid_map, TCudaGrid2D Grid)
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
__global__ void k_apply_map_stack_to_W_part_3(double *ws, double *x_y, TCudaGrid2D Grid, double *W_initial, int simulation_num)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid.NX || iY >= Grid.NY)
		return;

	long int In = iY*Grid.NX + iX;

	#ifndef DISCRET
		ws[In] = d_initial_W(x_y[2*In], x_y[2*In+1], simulation_num);
	#endif

	#ifdef DISCRET
		ws[In] = d_hermite_interpolate_2D(W_initial, x_y[2*In], x_y[2*In+1], Grid);
	#endif
}


// this function is nice, but not implemented for cpu mapstack yet
__global__ void k_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, TCudaGrid2D Grid_map, TCudaGrid2D Grid_fine, int simulation_num)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= Grid_fine.NX || iY >= Grid_fine.NY)
		return;

	int In = iY*Grid_fine.NX + iX;

	//position
	double x = iX*Grid_fine.hx;
	double y = iY*Grid_fine.hy;

	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, Grid_map);
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate_2D(&ChiX_stack[k*Grid_map.N*4], &ChiY_stack[k*Grid_map.N*4], x, y, &x, &y, Grid_map);

	error[In] = fabs(d_initial_W(x, y, simulation_num) - d_initial_W(iX*Grid_fine.hx, iY*Grid_fine.hy, simulation_num));
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




/*******************************************************************
*						 Initial condition						   *
*******************************************************************/

__device__ double d_initial_W(double x, double y, int simulation_num)
{
// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"

//	double ret = 0;
	switch (simulation_num) {
		case 0:  // 4_nodes
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			return cos(x) + cos(y) + 0.6*cos(2*x) + 0.2*cos(3*x);
			break;
		}
		case 1:  // quadropole
		{
			double ret = 0;
			for(int iy = -2; iy <= 2; iy++)
				for(int ix = -2; ix <= 2; ix++)
				{
					double dx = x - PI/2 + ix * 2*PI;
					double dy = y - PI/2 + iy * 2*PI;
					double A = 0.6258473;
					double s = 0.5;
					double B = A/(s*s*s*s) * (dx * dy) * (dx*dx + dy*dy - 6*s*s);
					double D = (dx*dx + dy*dy)/(2*s*s);
					ret += B * exp(-D);
				}
				return ret;
			break;
		}
		case 2:  // two vortices
		{
			double ret = 0;
			for(int iy = -1; iy <= 1; iy++)
				for(int ix = -1; ix <= 1; ix++)
				{
					ret += sin(0.5*(x + twoPI*ix))*sin(0.5*(x + twoPI*ix))*sin(0.5*((y + twoPI*iy) + twoPI*iy))*sin(0.5*((y + twoPI*iy) + twoPI*iy)) * (exp(-(((x + twoPI*ix) - PI)*((x + twoPI*ix) - PI) + ((y + twoPI*iy) - 0.33*twoPI)*((y + twoPI*iy) - 0.33*twoPI))*5) +
												exp(-(((x + twoPI*ix) - PI)*((x + twoPI*ix) - PI) + ((y + twoPI*iy) - 0.67*twoPI)*((y + twoPI*iy) - 0.67*twoPI))*5));		 //two votices of same size
				}
			return ret;
			break;
		}
		case 3:  // three vortices
		{
			//three vortices
			double ret = 0;
			double LX = PI/2;
			double LY = PI/(2.0*sqrt(2.0));

			for(int iy = -1; iy <= 1; iy++)
				for(int ix = -1; ix <= 1; ix++)
				{
					ret += sin(0.5*(x + twoPI*ix))*sin(0.5*(x + twoPI*ix))*sin(0.5*((y + twoPI*iy) + twoPI*iy))*sin(0.5*((y + twoPI*iy) + twoPI*iy)) *
								(
								+	exp(-(((x + twoPI*ix) - PI - LX)*((x + twoPI*ix) - PI - LX) + ((y + twoPI*iy) - PI)*((y + twoPI*iy) - PI))*5)
								+	exp(-(((x + twoPI*ix) - PI + LX)*((x + twoPI*ix) - PI + LX) + ((y + twoPI*iy) - PI)*((y + twoPI*iy) - PI))*5)
								-	exp(-(((x + twoPI*ix) - PI + LX)*((x + twoPI*ix) - PI + LX) + ((y + twoPI*iy) - PI - LY)*((y + twoPI*iy) - PI - LY))*5)
								);		 //two votices of same size
				}
			return ret;
			break;
		}
		case 4:  // single_shear_layer
		{
			//single shear layer
			double delta = 50;
			double delta2 = 0.01;
			double ret = 0;
			for(int iy = -1; iy <= 1; iy++)
				for(int iy = -1; iy <= 1; iy++)
					{
						ret +=    (1 + delta2 * cos(2*x))  *    exp( - delta * (y - PI) * (y - PI) );
					}
			ret /= 9;
			return ret;
			break;
		}
		case 5:  // turbulence_gaussienne
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			int NB_gaus = 8;		//NB_gaus = 6;sigma = 0.24;
			double sigma = 0.2;
			double ret = 0;
			for(int mu_x = 0; mu_x < NB_gaus; mu_x++){
				for(int mu_y = 0; mu_y < NB_gaus; mu_y++){
					ret += 1/(twoPI*sigma*sigma)*exp(-((x-mu_x*twoPI/(NB_gaus-1))*(x-mu_x*twoPI/(NB_gaus-1))/(2*sigma*sigma)+(y-mu_y*twoPI/(NB_gaus-1))*(y-mu_y*twoPI/(NB_gaus-1))/(2*sigma*sigma)));
				}
			}
			for(int mu_x = 0; mu_x < NB_gaus-1; mu_x++){
				for(int mu_y = 0; mu_y < NB_gaus-1; mu_y++){
					curandState_t state_x;
					curand_init((mu_x+1)*mu_y*mu_y, 0, 0, &state_x);
					double RAND_gaus_x = ((double)(curand(&state_x)%1000)-500)/100000;
					curandState_t state_y;
					curand_init((mu_y+1)*mu_x*mu_x, 0, 0, &state_y);
					double RAND_gaus_y = ((double)(curand(&state_y)%1000)-500)/100000;
					ret -= 1/(twoPI*sigma*sigma)*exp(-((x-(2*mu_x+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_x)*(x-(2*mu_x+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_x)/(2*sigma*sigma)+(y-(2*mu_y+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_y)*(y-(2*mu_y+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_y)/(2*sigma*sigma)));
				}
			}
			//curandState_t state;
			//curand_init(floor(y * 16384) * 16384 + floor(x * 16384), 0, 0, &state);
			//ret *= 1+((double)(curand(&state)%1000)-500)/100000;
			return ret - 0.008857380480028442;
			break;
		}
		// u(x,y)= - y 1/(nu^2 t^2) exp(-(x^2+y^2)/(4 nu t))
		// v(x,y)= + x 1/(nu^2 t^2) exp(-(x^2+y^2)/(4 nu t))
		case 6:  // shielded vortex
		{
			double nu = 2e-1;
			double nu_fac = 1 / (2*nu*nu);  // 1 / (2*nu*nu*nu)
			double nu_center = 4*nu;  // 4*nu
			double nu_scale = 4*nu;  // 4*nu

			// compute distance from center
			double x_r = PI-x; double y_r = PI-y;

			// build vorticity
			return nu_fac * (nu_center - x_r*x_r - y_r*y_r) * exp(-(x_r*x_r + y_r*y_r)/nu_scale);
			break;
		}
		default:  //default case goes to stationary
		{
			x = x - (x>0)*((int)(x/twoPI))*twoPI - (x<0)*((int)(x/twoPI)-1)*twoPI;
			y = y - (y>0)*((int)(y/twoPI))*twoPI - (y<0)*((int)(y/twoPI)-1)*twoPI;

			return cos(x)*cos(y);
			break;
		}
	}

}



__device__ double d_initial_W_discret(double x, double y, double *W_initial, int NX, int NY){
	// map position back into domain
	x = fmod(x, twoPI) - floor(x/twoPI)*twoPI;
	y = fmod(y, twoPI) - floor(y/twoPI)*twoPI;

	// compute index
	int In = floor(y/twoPI * NY) * NX + floor(x/twoPI * NX);
	//In = floor(x * NY) * NX + floor(y * NX);

	return W_initial[In];
}
