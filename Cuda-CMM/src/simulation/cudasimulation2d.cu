#include "cudasimulation2d.h"

#include <stdio.h>

////////////////////////////////////////////////////////////////////////
__global__ void kernel_init_diffeo(double *ChiX, double *ChiY, int NX, int NY, double h)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
		
	long int N = NX*NY;
	
	int In = iY*NX + iX;	
	
	ChiX[In] = iX*h;
	ChiY[In] = iY*h;
	
	// x dx = y dy = 1
	ChiX[1*N+In] = ChiY[2*N+In] = 1;
	
	// x dy = y dx = dxdy = 0
	ChiX[2*N+In] = ChiY[1*N+In] = ChiX[3*N+In] = ChiY[3*N+In] = 0;
}


__global__ void k_sample(double *ChiX, double *ChiY, double *ChiX_s, double *ChiY_s, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;	
	
	//position
	double x = iX*hs;
	double y = iY*hs;
	
	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	
	ChiX_s[In] = x;
	ChiY_s[In] = y;
}


__global__ void kernel_incompressibility_check(double *ChiX, double *ChiY, double *gradChi, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	//position
	double x = iX*hs + 0.5*hs;
	double y = iY*hs + 0.5*hs;
	
	gradChi[In] = device_diffeo_grad_2D(ChiX, ChiY, x, y, NXc, NYc, hc);
}


/*
 * Main advection function of the flow map using the stream function
 * Loop over footpoints to apply GALS
 * For each foot point: advect using the stream function and time stepping scheme
 * At the end: combine results of all footpoints using specific map update scheme
 */
__global__ void kernel_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *phi, int NXc, int NYc, double hc, int NX_psi, int NY_psi, double h_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num)			// time cost
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
    /*int NX = 512; //Dans l'interpolation remettre NXc Nyc hc
    int NY = 512;
    double h = twoPI/(float)NX;*/

    if(iX >= NXc || iY >= NYc)
		return;

	int In = iY*NXc + iX;

	long int N = NXc*NYc;

	//running through neighbours (unrolled loops)
	double x_ep[2], x_f[2];

	// initialize new intermediate values as zeros, helpfull to not write to array every point
	double Chi_full_x[4] = {0, 0, 0, 0};
	double Chi_full_y[4] = {0, 0, 0, 0};

	// factors on how the map will be updated in the end
	double c[3];
	switch (map_update_order_num) {
		// 6th order interpolation
		case 2: { c[0] = +3.0/8.0; c[1] = -3.0/20.0; c[2] = +1.0/40.0; break; }
		// 4th order interpolation
		case 1: { c[0] = +1.0/3.0; c[1] = -1.0/12.0; break; }
		// 2th order interpolation
		case 0: { c[0] = +1.0/4.0; break; }
		default: { c[0] = c[1] = c[2] = 0; break; }
	}

	// repeat for all footpoints, 4 for 2th order, 8 for 4th order and 12 for 6th order
	int k_total;
	switch (map_update_order_num) {
		case 2: { k_total = 12; break; }
		case 1: { k_total = 8; break; }
		case 0: { k_total = 4; break; }
	}

	// footpoint loop
	#pragma unroll
	for (int k_foot = 0; k_foot<k_total; k_foot++) {
		int i_foot_now = (k_foot/4);  // used for getting
		double i_dist_now = i_foot_now+1;

		// get position of footpoint, NE, SE, SW, NW
		// for higher orders repeat cross shape stencil with more points
		x_ep[0] = iX*hc + i_dist_now * ep*(1 - 2*((k_foot/2)%2));
		x_ep[1] = iY*hc + i_dist_now * ep*(1 - 2*(((k_foot+1)/2)%2));

		// time integration - note, we go backwards in time!
		switch (time_integration_num) {
			// EulerExp
			case 10: { euler_exp(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// ABTwo
			case 20: { adam_bashford_2_pc(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// ABTwo
			case 21: { RK2_heun(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// RKThree
			case 30: { RK3_classical(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// RKFour
			case 40: { RK4_classical(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// custom RKThree
			case 31: { RK3_optimized(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// custom RKFour case IV
			case 41: { RK4_optimized(phi, x_ep, x_f, NX_psi, NY_psi, h_psi, dt); break; }
			// zero on default
			default: { x_f[0] = x_f[1] = 0; }
		}

		// apply map deformation
		device_diffeo_interpolate_2D(ChiX, ChiY, x_f[0], x_f[1], &x_f[0], &x_f[1], NXc, NYc, hc);

		// directly apply map update
		// chi values - central average with stencil +NE +SE +SW +NW
		Chi_full_x[0] += x_f[0] * c[i_foot_now];
		Chi_full_y[0] += x_f[1] * c[i_foot_now];

		// chi grad x - central differences with stencil +NE +SE -SW -NW
		Chi_full_x[1] += x_f[0] * c[i_foot_now] * (1 - 2*((k_foot/2)%2)) /ep/i_dist_now;
		Chi_full_y[1] += x_f[1] * c[i_foot_now] * (1 - 2*((k_foot/2)%2)) /ep/i_dist_now;

		// chi grad y - central differences with stencil +NE -SE -SW +NW
		Chi_full_x[2] += x_f[0] * c[i_foot_now] * (1 - 2*(((k_foot+1)/2)%2)) /ep/i_dist_now;
		Chi_full_y[2] += x_f[1] * c[i_foot_now] * (1 - 2*(((k_foot+1)/2)%2)) /ep/i_dist_now;

		// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		Chi_full_x[3] += x_f[0] * c[i_foot_now] * (1 - 2*(k_foot%2)) /ep/ep/i_dist_now/i_dist_now;
		Chi_full_y[3] += x_f[1] * c[i_foot_now] * (1 - 2*(k_foot%2)) /ep/ep/i_dist_now/i_dist_now;
	}

	// can't use Chi because we still use it for diffeo_interpolate
	Chi_new_X[    In] = Chi_full_x[0];	Chi_new_Y[    In] = Chi_full_y[0];
	Chi_new_X[1*N+In] = Chi_full_x[1];	Chi_new_Y[1*N+In] = Chi_full_y[1];
	Chi_new_X[2*N+In] = Chi_full_x[2];	Chi_new_Y[2*N+In] = Chi_full_y[2];
	Chi_new_X[3*N+In] = Chi_full_x[3];	Chi_new_Y[3*N+In] = Chi_full_y[3];
}



/*******************************************************************
*						 Apply remapping						   *
*******************************************************************/


__global__ void kernel_apply_map_stack_to_W(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ws, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double *W_initial, int simulation_num)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	//position
	double x = iX*hs;
	double y = iY*hs;
	
	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate_2D(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x, y, simulation_num);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate_2D(W_initial, x, y, NXs, NYs, hs);
		//ws[In] = device_initial_W_discret(x, y, W_initial, NXs, NYs);
	#endif
	
}


__global__ void kernel_apply_map_stack_to_W_custom(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ws, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double xl, double xr, double yl, double yr, double *W_initial, int simulation_num)		// Zoom
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	double htemp = (xr - xl)/NXs;
	
	//position
	double x = xl + iX*htemp;
	double y = yl + iY*htemp;
	
	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate_2D(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);
		
	ws[In] = device_initial_W(x, y, simulation_num); //device_initial_W_discret(x, y)
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x, y, simulation_num);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate_2D(W_initial, x, y, NXs, NYs, hs);
		//ws[In] = device_initial_W_discret(x, y, W_initial, NXs, NYs);
	#endif
	
}


void apply_map_stack_to_W_part_All(TCudaGrid2D *Grid_fine, MapStack *Map_Stack, double *ChiX, double *ChiY,
		double *W_real, double *Dev_Temp, double *bounds, double *W_initial, int simulation_num)
{
	// for normal map stack, bounds has the domain boundaries applied
	kernel_apply_map_stack_to_W_custom_part_1<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX, ChiY, Dev_Temp, Map_Stack->Grid->NX, Map_Stack->Grid->NY, Map_Stack->Grid->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, bounds[0], bounds[1], bounds[2], bounds[3]);

	// loop over all maps in map stack
	for(int K_RAM = Map_Stack->stack_length_Nb_array_RAM; K_RAM >= 0; K_RAM--){
		if (K_RAM == Map_Stack->stack_length_Nb_array_RAM){
			for(int K = Map_Stack->stack_length_RAM%Map_Stack->frac_mem_cpu_to_gpu; K >= 0; K--){
				// copy map stack, is done intern in map stack class
				Map_Stack->copy_map_to_device(K_RAM, K);

				if (K == Map_Stack->stack_length_RAM%Map_Stack->frac_mem_cpu_to_gpu){
					for(int k = Map_Stack->map_stack_ctr - Map_Stack->stack_length_RAM*Map_Stack->map_stack_length - 1; k >= 0; k--){
						kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Map_Stack->Dev_ChiX_stack, Map_Stack->Dev_ChiY_stack, Dev_Temp, Map_Stack->Grid->NX, Map_Stack->Grid->NY, Map_Stack->Grid->h, Grid_fine->NX, Grid_fine->NY, k);
					}
				}
				else{
					for(int k = Map_Stack->map_stack_length - 1; k >= 0; k--){
						kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Map_Stack->Dev_ChiX_stack, Map_Stack->Dev_ChiY_stack, Dev_Temp, Map_Stack->Grid->NX, Map_Stack->Grid->NY, Map_Stack->Grid->h, Grid_fine->NX, Grid_fine->NY, k);
					}
				}
			}
		}
		else{
			for(int K = Map_Stack->frac_mem_cpu_to_gpu-1; K >= 0; K--){
				Map_Stack->copy_map_to_device(K_RAM, K);
				for(int k = Map_Stack->map_stack_length - 1; k >= 0; k--){
					kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Map_Stack->Dev_ChiX_stack, Map_Stack->Dev_ChiY_stack, Dev_Temp, Map_Stack->Grid->NX, Map_Stack->Grid->NY, Map_Stack->Grid->h, Grid_fine->NX, Grid_fine->NY, k);
				}
			}
		}
	}

	// initial condition
	kernel_apply_map_stack_to_W_part_3<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Temp, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial, simulation_num);
}


__global__ void kernel_apply_map_stack_to_W_custom_part_1(double *ChiX, double *ChiY, double *x_y, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double xl, double xr, double yl, double yr)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	double htemp = (xr - xl)/NXs;

	//position
	double x = xl + iX*htemp;
	double y = yl + iY*htemp;
	
	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	
	// save in two points in array
	x_y[2*In  ] = x;
	x_y[2*In+1] = y;

}

__global__ void kernel_apply_map_stack_to_W_part_2(double *ChiX_stack, double *ChiY_stack, double *x_y, int NXc, int NYc, double hc, int NXs, int NYs, int k)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	//for(int k = stack_length - 1; k >= 0; k--)
	device_diffeo_interpolate_2D(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x_y[2*In], x_y[2*In+1], &x_y[2*In], &x_y[2*In+1], NXc, NYc, hc);
	
}

__global__ void kernel_apply_map_stack_to_W_part_3(double *ws, double *x_y, int NXs, int NYs, double hs, double *W_initial, int simulation_num)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x_y[2*In], x_y[2*In+1], simulation_num);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate_2D(W_initial, x_y[2*In], x_y[2*In+1], NXs, NYs, hs);
	#endif
	
}


/*******************************************************************
*						 										   *
*******************************************************************/


__global__ void kernel_compare_vorticity_with_initial(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs, int simulation_num)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	//position
	double x = iX*hs;
	double y = iY*hs;
	
	device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate_2D(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);
	
	error[In] = fabs(device_initial_W(x, y, simulation_num) - device_initial_W(iX*hs, iY*hs, simulation_num));
}


// apply mollifier
__global__ void kernel_apply_map_and_sample_from_hermite(double *ChiX, double *ChiY, double *fs, double *H, int NXc, int NYc, double hc, int NXs, int NYs, double hs, int NXh, int NYh, double hh, int molly_stencil)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);

	if(iX >= NXs || iY >= NYs)
		return;

	int In = iY*NXs + iX;

	//position
	double x = iX*hs;
	double y = iY*hs;

	// mollification to act as a lowpass filter
	double x2, y2;

	/*                                      0  1/6  0
	 * mollifier of order 1 using stencil  1/6 1/3 1/6
	 * using 4 neighbouring points			0  1/6  0
	 */
	if (molly_stencil == 4) {
		// compute main points
		device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x2, &y2, NXc, NYc, hc);
		double moll_add = device_hermite_interpolate_2D(H, x2, y2, NXh, NYh, hh)/3.0;  // other values will be added on here
		for (int i_molly = 0; i_molly < 4; i_molly++) {
			// choose 4 points in between the grid: W, E, S, N
			x2 = x + hs/2.0*((i_molly/2+1)%2) * (-1 + 2*(i_molly%2));  // -1 +1  0  0
			y2 = y + hs/2.0*((i_molly/2  )%2) * (-1 + 2*(i_molly%2));  //  0  0 -1 +1

			device_diffeo_interpolate_2D(ChiX, ChiY, x2, y2, &x2, &y2, NXc, NYc, hc);
			moll_add += device_hermite_interpolate_2D(H, x2, y2, NXh, NYh, hh)/6.0;
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
			x2 = x + hs*(-1 + i_molly%3)/2.0;
			y2 = y + hs*(-1 + i_molly/3)/2.0;

			device_diffeo_interpolate_2D(ChiX, ChiY, x2, y2, &x2, &y2, NXc, NYc, hc);
			moll_add += (1 + (i_molly%3)%2) * (1 + (i_molly/3)%2) * device_hermite_interpolate_2D(H, x2, y2, NXh, NYh, hh)/16.0;
		}
		fs[In] = moll_add;
	}
	// else, assume no mollification
	else {
		double x2, y2;
		device_diffeo_interpolate_2D(ChiX, ChiY, x, y, &x2, &y2, NXc, NYc, hc);
		fs[In] = device_hermite_interpolate_2D(H, x2, y2, NXh, NYh, hh);
	}
}




/*******************************************************************
*						 Initial condition						   *
*******************************************************************/

__device__ double device_initial_W(double x, double y, int simulation_num)
{
// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"

//	double ret = 0;
	switch (simulation_num) {
		case 0:  // 4_nodes
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;
//			x += x - (x>0)*((int)(x/twoPI))*twoPI - (x<0)*((int)(x/twoPI)-1)*twoPI;
//			y += y - (y>0)*((int)(y/twoPI))*twoPI - (y<0)*((int)(y/twoPI)-1)*twoPI;

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
//			x = fmod(x, twoPI);
//			x = (x < 0)*(twoPI+x) + (x > 0)*(x);
//			y = fmod(y, twoPI);
//			y = (y < 0)*(twoPI+y) + (y > 0)*(y);
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
			double nu = 5e-1;
			double t = 1;

			// compute distance from center
			double x_r = PI-x; double y_r = PI-y;

			// build vorticity
			return (4*nu*t - x_r*x_r - y_r*y_r) / (2*nu*nu*nu*t*t*t) * exp(-(x_r*x_r + y_r*y_r)/(4*nu*t));
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



__device__ double device_initial_W_discret(double x, double y, double *W_initial, int NX, int NY){
	// map position back into domain
	x = fmod(x, twoPI);
	x -= floor(x/twoPI)*twoPI;
//	x = (x < 0)*(twoPI+x) + (x > 0)*(x);
	y = fmod(y, twoPI);
	y -= floor(y/twoPI)*twoPI;
//	y = (y < 0)*(twoPI+y) + (y > 0)*(y);
	
	// compute index
	int In = floor(y/twoPI * NY) * NX + floor(x/twoPI * NX);
	//In = floor(x * NY) * NX + floor(y * NX); 
	
	return W_initial[In];
}








///*******************************************************************
//*							   Zoom								   *
//*******************************************************************/
//
//
//void Zoom(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *Dev_ChiX_stack, double *Dev_ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *Dev_ChiX, double *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, double *W_real, cufftHandle cufftPlan_fine, double *W_initial, cufftDoubleComplex *Dev_Complex_fine, string workspace, string simulationName, int simulation_num, double L)
//{
//	double *ws;
//	ws = new double[Grid_fine->N];
//	int save_ctr = 0;
//
//	double xCenter = 0.54;
//	double yCenter = 0.51;
//	double width = 0.5;
//
//	double xMin = xCenter - width/2;
//	double xMax = xMin + width;
//	double yMin = yCenter - width/2;
//	double yMax = yMin + width;
//
//	std::ostringstream ss;
//	ss<<save_ctr;
//
//
//	//save zooming effects
//	for(int zoom_ctr = 0; zoom_ctr<10; zoom_ctr++){
//
//		width *=  0.5;//0.99
//		xMin = xCenter - width/2;
//		xMax = xMin + width;
//		yMin = yCenter - width/2;
//		yMax = yMin + width;
//
//
//		//kernel_apply_map_stack_to_W_custom<<<Gsf->blocksPerGrid, Gsf->threadsPerBlock>>>(devChiX_stack, devChiY_stack, devChiX, devChiY, devWs, stack_map_passed, Gc->NX, Gc->NY, Gc->h, Gsf->NX, Gsf->NY, Gsf->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial);
//		kernel_apply_map_stack_to_W_custom_part_All(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, W_real, Dev_Complex_fine, stack_length, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial, simulation_num);
//
//
//		cudaMemcpy(ws, W_real, Grid_fine->sizeNReal, cudaMemcpyDeviceToHost);
//
//		std::ostringstream ss2;
//		ss2<<zoom_ctr;
//
//		writeAllRealToBinaryFile(Grid_fine->N, ws, simulationName, workspace, "zoom_" + ss2.str());
//	}
//
//}


/*******************************************************************
*							    								   *
*******************************************************************/



















/******************************************************************/
/*******************************************************************
*							   Old								   *
*******************************************************************/
/******************************************************************/












//////////////////////////////////////////////////////////////////////////
//void test_fft_operations()
//{
//}
//
////void recompute_output_files(){}
//
//////////////////////////////////////////////////////////////////////////
//double compare_map_with_identity(double *chiX, double *chiY, int NX, int NY, double h)
//{
//return 0;
//}
//
//__global__ void kernel_compute_total_grid_Chi(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *gradChi, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
//{
//}
//
//__global__ void kernel_compute_enstropy_increase_rate_factors(double *w, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, double ep)
//{
//}
//
//__global__ void kernel_compute_enstropy_increase_rate_factors(double *wHsc, double *ChiX, double *ChiY, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, int NXsc, int NYsc, double hsc, double ep)
//{
//}
//
//////////////////////////////////////////////////////////////////////////
//__global__ void kernel_advect_using_velocity_function(double *ChiX, double *ChiY, double *ChiDualX, double *ChiDualY,  int NXc, int NYc, double hc, double t, double dt, double ep)
//{
//}
//
//////////////////////////////////////////////////////////////////////////
//__global__ void kernel_apply_map_to_W(double *ChiX, double *ChiY, double *ws, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
//{
//}
//
//__global__ void kernel_compare_map_stack_with_identity(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
//{
//}




























