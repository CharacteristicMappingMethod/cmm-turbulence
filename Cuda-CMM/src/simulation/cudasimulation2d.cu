#include "cudasimulation2d.h"


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
	
	ChiX[1*N+In] = ChiY[2*N+In] = 1; 
	
	ChiX[2*N+In] = ChiY[1*N+In] = 
	ChiX[3*N+In] = ChiY[3*N+In] = 0;
}


__global__ void upsample(double *ChiX, double *ChiY, double *ChiX_2048, double *ChiY_2048, int NXc, int NYc, double hc, int NXs, int NYs, double hs)															// time cost
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
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);		
	
	ChiX_2048[In] = x;
	ChiY_2048[In] = y;
}


void Psi_upsampling(TCudaGrid2D *Grid_2048, double *Dev_W_2048, cufftDoubleComplex *Dev_Complex_fine_2048, cufftDoubleComplex *Dev_Hat_fine_bis_2048, cufftDoubleComplex *Dev_Hat_fine_2048, double *Dev_Psi_2048, cufftHandle cufftPlan_2048){

	kernel_real_to_complex<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_W_2048, Dev_Complex_fine_2048, Grid_2048->NX, Grid_2048->NY);
	cufftExecZ2Z(cufftPlan_2048, Dev_Complex_fine_2048, Dev_Hat_fine_bis_2048, CUFFT_FORWARD);
	kernel_normalize<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Grid_2048->NX, Grid_2048->NY);

	// Forming Psi hermite
	kernel_fft_iLap<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Grid_2048->NX, Grid_2048->NY, Grid_2048->h);													// Inverse laplacian in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_Psi_2048, Dev_Complex_fine_2048, Grid_2048->NX, Grid_2048->NY);

	kernel_fft_dy<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_Hat_fine_2048, Dev_Hat_fine_bis_2048, Grid_2048->NX, Grid_2048->NY, Grid_2048->h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_bis_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(&Dev_Psi_2048[2*Grid_2048->N], Dev_Complex_fine_2048, Grid_2048->NX, Grid_2048->NY);

	kernel_fft_dx<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_Hat_fine_2048, Dev_Hat_fine_bis_2048, Grid_2048->NX, Grid_2048->NY, Grid_2048->h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_bis_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(&Dev_Psi_2048[Grid_2048->N], Dev_Complex_fine_2048, Grid_2048->NX, Grid_2048->NY);

	kernel_fft_dy<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Grid_2048->NX, Grid_2048->NY, Grid_2048->h);													// y-derivative of x-derivative of of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real<<<Grid_2048->blocksPerGrid, Grid_2048->threadsPerBlock>>>(&Dev_Psi_2048[3*Grid_2048->N], Dev_Complex_fine_2048, Grid_2048->NX, Grid_2048->NY);

}


__global__ void kernel_incompressibility_check(double *ChiX, double *ChiY, double *gradChi, int NXc, int NYc, double hc, int NXs, int NYs, double hs)															// time cost
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
	
	gradChi[In] = device_diffeo_grad(ChiX, ChiY, x, y, NXc, NYc, hc);
}


/*
 * Main advection function of the flow map using the stream function
 * Loop over footpoints to apply GALS
 * For each foot point: advect using the stream function and time stepping scheme
 * At the end: combine results of all footpoints using specific map update scheme
 */
__global__ void kernel_advect_using_stream_hermite(double *ChiX, double *ChiY, double *Chi_new_X, double *Chi_new_Y, double *phi, double *phi_p, double *phi_p_p, int NXc, int NYc, double hc, int NX_psi, int NY_psi, double h_psi, double t, double dt, double ep, int time_integration_num, int map_update_order_num)			// time cost
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
	double xep, yep;
    double xf, yf;  // coordinates in loop
	double xf_p, yf_p;  // variables used for fixed-point iteration in ABTwo

	// initialize new intermediate values as zeros, helpfull to not write to array every point
	double Chi_full_x[4] = {0, 0, 0, 0};
	double Chi_full_y[4] = {0, 0, 0, 0};

	double u, v, u_p, v_p, u_p_p, v_p_p;  // velocity at current and previous time steps
	double k1_x, k1_y, k2_x, k2_y, k3_x, k3_y; // different intermediate functions for RKThree

	// factors on how the map will be updated in the end
	double c1[3], c2[3], c3[3];
	// 4th order interpolation
	if (map_update_order_num == 2) {
		// 4th order values factors for inner, middle and outer points
		c1[0] = +3.0/(8.0); 			c1[1] = -3.0/(20.0);			c1[2] = +1.0/(40.0);  // 0th order
		c2[0] = +29.0/(252.0)/ep; 		c2[1] = +67.0/(504.0)/ep;		c2[2] = -11.0/(252.0)/ep;  // 1th order
		c3[0] = +251.0/(6288.0)/ep/ep;	c3[1] = +607.0/(6288.0)/ep/ep;	c3[2] = -41.0/(2096.0)/ep/ep; // 1th order cross
	}
	// 3rd order interpolation
	else if (map_update_order_num == 1) {
		// 3rd order values factors for inner and outer points
		c1[0] = +1.0/(3.0); 		c1[1] = -1.0/(12.0);  // 0th order
		c2[0] = +1.0/(3.0)/ep; 		c2[1] = -1.0/(24.0)/ep;  // 1th order
		c3[0] = +1.0/(68.0)/ep/ep;	c3[1] = +1.0/(17.0)/ep/ep;  // 1th order cross
	}
	// 2nd order interpolation
	else if (map_update_order_num == 0) {
		// 2rd order values factors for middle points
		c1[0] = +1.0/(4.0);  // 0th order
		c2[0] = +1.0/(4.0)/ep;  // 0th order
		c3[0] = +1.0/(4.0)/ep/ep;  // 0th order
	}
	// zero
	else {
		c1[0] = c1[1] = c1[2] = c2[0] = c2[1] = c2[2] = c3[0] = c3[1] = c3[2] = 0;
	}

	// repeat for all footpoints, 4 for 3th order, 8 for 4th order and 12 for 5th order
	int k_total;
	if (map_update_order_num == 2) {
		k_total = 12;
	}
	else if (map_update_order_num == 1) {
		k_total = 8;
	}
	else k_total = 4;

	// footpoint loop
	for (int k_foot = 0; k_foot<k_total; k_foot++) {
		// get position of footpoint, NE, SE, SW, NW
		// for higher orders repeat cross shape stencil with more points
		xep = iX*hc + (1 + k_foot/4) * ep*(1 - 2*((k_foot/2)%2));
		yep = iY*hc + (1 + k_foot/4) * ep*(1 - 2*(((k_foot+1)/2)%2));

		xf = xep;
		yf = yep;

		switch (time_integration_num) {
			case 0:  // EulerExp
			{
				device_hermite_interpolate_dx_dy_1(phi, xf, yf, &u, &v, NX_psi, NY_psi, h_psi);
				xf = xep - dt * u;
				yf = yep - dt * v;
				break;
			}
			// Adam-Bashfords of order two, where a fixed point iteration with 10 iterations
			case 1:  // ABTwo
			{
				xf_p = xep; yf_p = yep;

				// fixed point iteration for xf,yf using previous foot points (self-correction)
	            #pragma unroll 10
				for(int ctr = 0; ctr<10; ctr++)
	            {
					//step 1
	                device_hermite_interpolate_dx_dy_1(phi_p, xf_p, yf_p, &u_p, &v_p, NX_psi, NY_psi, h_psi);

	                xf_p = xf - dt * u_p;
	                yf_p = yf - dt * v_p;

					//step 2
	                device_hermite_interpolate_dx_dy_1(phi_p, xf_p, yf_p, &u_p, &v_p, NX_psi, NY_psi, h_psi);
	                device_hermite_interpolate_dx_dy_1(phi, xf , yf, &u, &v, NX_psi, NY_psi, h_psi);

	                xf = xep - dt * (1.5*u - 0.5*u_p);
	                yf = yep - dt * (1.5*v - 0.5*v_p);

	            }
	            break;
			}
			// RKThree time step utilizing some intermediate steps
			case 2:  // RKThree
			{
				// compute u_tilde(X,t_n+1)
				device_hermite_interpolate_dx_dy_3(phi_p_p, phi_p, phi, xep, yep, &u_p_p, &v_p_p, &u_p, &v_p, &u, &v, NX_psi, NY_psi, h_psi);

				// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
				k1_x = 3 * u + -3 * u_p + u_p_p;
				k1_y = 3 * v + -3 * v_p + v_p_p;

				// compute u_tilde(x - dt*k1/2, t_n+1 - dt/2)
				device_hermite_interpolate_dx_dy_3(phi_p_p, phi_p, phi, xep - dt*k1_x/2, yep - dt*k1_y/2, &u_p_p, &v_p_p, &u_p, &v_p, &u, &v, NX_psi, NY_psi, h_psi);

				//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 1.875*u_n - 1.25*u_n-1 + 0.375*u_n-2
				k2_x = 1.875 * u + -1.25 * u_p + 0.375 * u_p_p;
				k2_y = 1.875 * v + -1.25 * v_p + 0.375 * v_p_p;

				//compute u_tilde(x + dt * k1 - 2*dt*k2, t_n+1 - dt)
				device_hermite_interpolate_dx_dy_1(phi, xep + dt*k1_x - 2*dt*k2_x, yep + dt*k1_y - 2*dt*k2_y, &u, &v, NX_psi, NY_psi, h_psi);

				// k3 = u_tilde(x = k1 dt - 2 k2 dt, t_n) = u

				// build all RK-steps together
				xf = xep - dt * (k1_x + 4*k2_x + u)/6;
				yf = yep - dt * (k1_y + 4*k2_y + v)/6;
				break;
			}
			// RKFour time step utilizing some intermediate steps
			case 3:  // RKFour
			{
				// compute u_tilde(X,t_n+1)
				device_hermite_interpolate_dx_dy_3(phi_p_p, phi_p, phi, xep, yep, &u_p_p, &v_p_p, &u_p, &v_p, &u, &v, NX_psi, NY_psi, h_psi);

				// k1 = u_tilde(x,t_n+1) = 3*u_n - 3*u_n-1 + 1*u_n-2
				k1_x = 3 * u + -3 * u_p + u_p_p;
				k1_y = 3 * v + -3 * v_p + v_p_p;

				// compute u_tilde(x - dt*k1/2, t_n+1 - dt/2)
				device_hermite_interpolate_dx_dy_3(phi_p_p, phi_p, phi, xep - dt*k1_x/2, yep - dt*k1_y/2, &u_p_p, &v_p_p, &u_p, &v_p, &u, &v, NX_psi, NY_psi, h_psi);

				//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2) = 1.875*u_n - 1.25*u_n-1 + 0.375*u_n-2
				k2_x = 1.875 * u + -1.25 * u_p + 0.375 * u_p_p;
				k2_y = 1.875 * v + -1.25 * v_p + 0.375 * v_p_p;

				// compute u_tilde(x - dt*k2/2, t_n+1 - dt/2)
				device_hermite_interpolate_dx_dy_3(phi_p_p, phi_p, phi, xep - dt*k2_x/2, yep - dt*k2_y/2, &u_p_p, &v_p_p, &u_p, &v_p, &u, &v, NX_psi, NY_psi, h_psi);

				//k2 = u_tilde(x - k2 dt/2, t_n+1 - dt/2) = 1.875*u_n - 1.25*u_n-1 + 0.375*u_n-2
				k3_x = 1.875 * u + -1.25 * u_p + 0.375 * u_p_p;
				k3_y = 1.875 * v + -1.25 * v_p + 0.375 * v_p_p;

				//compute u_tilde(x - dt*k3, t_n+1 - dt)
				device_hermite_interpolate_dx_dy_1(phi, xep - dt*k3_x, yep - dt*k3_y, &u, &v, NX_psi, NY_psi, h_psi);

				// k3 = u_tilde(x - k3 dt, t_n) = u

				// build all RK-steps together
				xf = xep - dt * (k1_x + 2*k2_x + 2*k3_x + u)/6;
				yf = yep - dt * (k1_y + 2*k2_y + 2*k3_y + v)/6;
				break;
			}
			default:  // EulerExp on default
			{
				device_hermite_interpolate_dx_dy_1(phi, xf, yf, &u, &v, NX_psi, NY_psi, h_psi);
				xf = xep - dt * u; yf = yep - dt * v;
				break;
			}
		}
		// apply map?
		device_diffeo_interpolate(ChiX, ChiY, xf, yf, &xf, &yf, NXc, NYc, hc);

		// directly apply map update
		// chi values - central average with stencil +NE +SE +SW +NW
		Chi_full_x[0] += xf * c1[k_foot/4];
		Chi_full_y[0] += yf * c1[k_foot/4];

		// chi grad x - central differences with stencil +NE +SE -SW -NW
		Chi_full_x[1] += xf * c2[k_foot/4] * (1 - 2*((k_foot/2)%2));
		Chi_full_y[1] += yf * c2[k_foot/4] * (1 - 2*((k_foot/2)%2));

		// chi grad y - central differences with stencil +NE -SE -SW +NW
		Chi_full_x[2] += xf * c2[k_foot/4] * (1 - 2*(((k_foot+1)/2)%2));
		Chi_full_y[2] += yf * c2[k_foot/4] * (1 - 2*(((k_foot+1)/2)%2));

		// chi grad x y - cross central differences with stencil +NE -SE +SW -NW
		Chi_full_x[3] += xf * c3[k_foot/4] * (1 - 2*(k_foot%2));
		Chi_full_y[3] += yf * c3[k_foot/4] * (1 - 2*(k_foot%2));
	}

	// can't use Chi because we still use it for diffeo_interpolate
	Chi_new_X[    In] = Chi_full_x[0];	Chi_new_Y[    In] = Chi_full_y[0];
	Chi_new_X[1*N+In] = Chi_full_x[1];	Chi_new_Y[1*N+In] = Chi_full_y[1];
	Chi_new_X[2*N+In] = Chi_full_x[2];	Chi_new_Y[2*N+In] = Chi_full_y[2];
	Chi_new_X[3*N+In] = Chi_full_x[3];	Chi_new_Y[3*N+In] = Chi_full_y[3];
}


/*******************************************************************
*						 Real and Complex						   *
*******************************************************************/
	
__global__ void kernel_real_to_complex(double *varR, cufftDoubleComplex *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	varC[In].x = varR[In];
	varC[In].y = 0.0;
}


__global__ void kernel_complex_to_real(double *varR, cufftDoubleComplex *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	varR[In] = varC[In].x;
}


__global__ void kernel_real_to_complex_H(double *varR, cufftDoubleComplex *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	long int N = NX*NY;
	
	varC[In].x = varR[In];
	varC[N+In].x = varR[N+In];
	varC[2*N+In].x = varR[2*N+In];
	varC[3*N+In].x = varR[3*N+In];
	varC[In].y = varC[N+In].y = varC[2*N+In].y = varC[3*N+In].y = 0.0;
	
}


__global__ void kernel_complex_to_real_H(double *varR, cufftDoubleComplex *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	long int N = NX*NY;
	
	varR[In] = varC[In].x;
	varR[N+In] = varC[N+In].x;
	varR[2*N+In] = varC[2*N+In].x;
	varR[3*N+In] = varC[3*N+In].x;
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
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);		
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);		
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x, y, simulation_num);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate(W_initial, x, y, NXs, NYs, hs);
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
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);
		
	ws[In] = device_initial_W(x, y, simulation_num); //device_initial_W_discret(x, y)
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x, y, simulation_num);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate(W_initial, x, y, NXs, NYs, hs);
		//ws[In] = device_initial_W_discret(x, y, W_initial, NXs, NYs);
	#endif
	
}

/*
void kernel_apply_map_stack_to_W_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ChiX_stack_RAM, double *ChiY_stack_RAM, double *W_real, cufftDoubleComplex *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double *W_initial)
{
	
	kernel_apply_map_stack_to_W_part_1<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX, ChiY, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
	for(int K = stack_length_RAM; K >= 0; K--){
	 	cudaMemcpy(ChiX_stack, &ChiX_stack_RAM[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
		cudaMemcpy(ChiY_stack, &ChiY_stack_RAM[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
		if (K == stack_length_RAM){
			for(int k = stack_length - stack_length_RAM*map_stack_length - 1; k >= 0; k--){
				kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
			}
		}
		else{
			for(int k = map_stack_length - 1; k >= 0; k--){
				kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
			}
		}
	}
	kernel_apply_map_stack_to_W_part_3<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
}*/


void kernel_apply_map_stack_to_W_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *W_real, cufftDoubleComplex *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double *W_initial, int simulation_num)
{
	
	kernel_apply_map_stack_to_W_part_1<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX, ChiY, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
	
	for(int K_RAM = stack_length_Nb_array_RAM; K_RAM >= 0; K_RAM--){
		if (K_RAM == stack_length_Nb_array_RAM){
			for(int K = stack_length_RAM%mem_RAM; K >= 0; K--){
				if (K_RAM == 0){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 1){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 2){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 3){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K == stack_length_RAM%mem_RAM){
					for(int k = stack_length - stack_length_RAM*map_stack_length - 1; k >= 0; k--){
						kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
					}
				}
				else{
					for(int k = map_stack_length - 1; k >= 0; k--){
						kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
					}
				}
			}
		}
		else{
			for(int K = mem_RAM-1; K >= 0; K--){
				if (K_RAM == 0){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 1){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 2){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 3){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				for(int k = map_stack_length - 1; k >= 0; k--){
					kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
				}
			}
		}
	}
	
	kernel_apply_map_stack_to_W_part_3<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial, simulation_num);
	
}


__global__ void kernel_apply_map_stack_to_W_part_1(double *ChiX, double *ChiY, cufftDoubleComplex *x_y, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
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
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	
	x_y[In].x = x;
	x_y[In].y = y;
	
}

__global__ void kernel_apply_map_stack_to_W_part_2(double *ChiX_stack, double *ChiY_stack, cufftDoubleComplex *x_y, int NXc, int NYc, double hc, int NXs, int NYs, int k)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	//for(int k = stack_length - 1; k >= 0; k--)
	device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x_y[In].x, x_y[In].y, &x_y[In].x, &x_y[In].y, NXc, NYc, hc);
	
}

__global__ void kernel_apply_map_stack_to_W_part_3(double *ws, cufftDoubleComplex *x_y, int NXs, int NYs, double hs, double *W_initial, int simulation_num)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x_y[In].x, x_y[In].y, simulation_num);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate(W_initial, x_y[In].x, x_y[In].y, NXs, NYs, hs);
	#endif
	
}


void kernel_apply_map_stack_to_W_custom_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *W_real, cufftDoubleComplex *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double xl, double xr, double yl, double yr, double *W_initial, int simulation_num)
{
	
	kernel_apply_map_stack_to_W_custom_part_1<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX, ChiY, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, xl, xr, yl, yr);
	
	for(int K_RAM = stack_length_Nb_array_RAM; K_RAM >= 0; K_RAM--){
		if (K_RAM == stack_length_Nb_array_RAM){
			for(int K = stack_length_RAM%mem_RAM; K >= 0; K--){
				if (K_RAM == 0){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 1){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 2){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 3){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K == stack_length_RAM%mem_RAM){
					for(int k = stack_length - stack_length_RAM*map_stack_length - 1; k >= 0; k--){
						kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
					}
				}
				else{
					for(int k = map_stack_length - 1; k >= 0; k--){
						kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
					}
				}
			}
		}
		else{
			for(int K = mem_RAM-1; K >= 0; K--){
				if (K_RAM == 0){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_0[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 1){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_1[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 2){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_2[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				if (K_RAM == 3){
					cudaMemcpy(ChiX_stack, &Host_ChiX_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
					cudaMemcpy(ChiY_stack, &Host_ChiY_stack_RAM_3[K*map_stack_length*Grid_coarse->N*4], map_stack_length * 4*Grid_coarse->sizeNReal, cudaMemcpyHostToDevice);
				}
				for(int k = map_stack_length - 1; k >= 0; k--){
					kernel_apply_map_stack_to_W_part_2<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(ChiX_stack, ChiY_stack, Dev_Complex_fine, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, k);
				}
			}
		}
	}
	
	kernel_apply_map_stack_to_W_part_3<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial, simulation_num);
	
}


__global__ void kernel_apply_map_stack_to_W_custom_part_1(double *ChiX, double *ChiY, cufftDoubleComplex *x_y, int NXc, int NYc, double hc, int NXs, int NYs, double hs, double xl, double xr, double yl, double yr)
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
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	
	x_y[In].x = x;
	x_y[In].y = y;
	
}


__global__ void cut_off_scale(cufftDoubleComplex *W, int NX)
{
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	int In = iY*NX + iX;
	
	int i = In/NX;
	int j = In%NX;
	if (i <= NX/2 && j <= NX/2)
		if ((i*i + j*j) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	if (i > NX/2 && j <= NX/2)
		if (((NX-i)*(NX-i) + j*j) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	if (i <= NX/2 && j > NX/2)
		if ((i*i + (NX-j)*(NX-j)) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	if (i > NX/2 && j > NX/2)
		if (((NX-i)*(NX-i) + (NX-j)*(NX-j)) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	
	if (In == 0){
		W[In].x = 0;
		W[In].y = 0;
	}
	
}


// for zero padding we need to move all outer elements to conserve symmetry of spectrum
__global__ void zero_move(cufftDoubleComplex *In, cufftDoubleComplex *Out, double Nc, double Ns) {
	int iXc = (blockDim.x * blockIdx.x + threadIdx.x);
	int iYc = (blockDim.y * blockIdx.y + threadIdx.y);

	int Inc = iYc*Nc + iXc;

	// get new positions
	int iXs = iXc;
	int iYs = iYc;
	// shift upper half to the outer edge
	if (iXc > Nc/2) {
		iXs += Ns - Nc;
	}
	if (iYc > Nc/2) {
		iYs += Ns - Nc;
	}
	// get index in new system
	int Ins = iYs*Ns + iXs;
	// transcribe
	Out[Ins].x = In[Inc].x;
	Out[Ins].x = In[Inc].x;
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
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);		
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);		
	
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
		device_diffeo_interpolate(ChiX, ChiY, x, y, &x2, &y2, NXc, NYc, hc);
		double moll_add = device_hermite_interpolate(H, x2, y2, NXh, NYh, hh)/3.0;  // other values will be added on here
		for (int i_molly = 0; i_molly < 4; i_molly++) {
			// choose 4 points in between the grid: W, E, S, N
			double x2 = x + hs/2*((i_molly/2+1)%2) * (-1 + 2*(i_molly%2));  // -1 +1  0  0
			double y2 = y + hs/2*((i_molly/2  )%2) * (-1 + 2*(i_molly%2));  //  0  0 -1 +1

			device_diffeo_interpolate(ChiX, ChiY, x2, y2, &x2, &y2, NXc, NYc, hc);
			moll_add += device_hermite_interpolate(H, x2, y2, NXh, NYh, hh)/6.0;
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
			double x2 = x + hs*(-1 + i_molly%3)/2;
			double y2 = y + hs*(-1 + i_molly/3)/2;

			device_diffeo_interpolate(ChiX, ChiY, x2, y2, &x2, &y2, NXc, NYc, hc);
			moll_add += (1 + (i_molly%3)%2) * (1 + (i_molly/3)%2) * device_hermite_interpolate(H, x2, y2, NXh, NYh, hh)/16.0;
		}
		fs[In] = moll_add;
	}
	// else, assume no mollification
	else {
		device_diffeo_interpolate(ChiX, ChiY, x, y, &x2, &y2, NXc, NYc, hc);
		fs[In] = device_hermite_interpolate(H, x2, y2, NXh, NYh, hh);
	}
}


/*******************************************************************
*						 										   *
*******************************************************************/

////////////////////////////////////////////////////////////////////////
__global__ void kernel_sample_on_coarse_grid(cufftDoubleComplex *AcOut, cufftDoubleComplex *AfOut, int NXc, int NYc, double hc, int NXf, int NYf, double hf)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXc || iY >= NYc)
		return;
	
	int In = iY*NXc + iX;	
		
		////////////////////////////////////////////////////
		//same sampling grid
		AcOut[In].x = AfOut[In].x;
		AcOut[In].y = AfOut[In].y;
		return;
	
}


__global__ void kernel_normalize(cufftDoubleComplex *F, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	double N = NX*NY;
	
	F[In].x /= (double)N;
	F[In].y /= (double)N;
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
			x = x - (x>0)*((int)(x/twoPI))*twoPI - (x<0)*((int)(x/twoPI)-1)*twoPI;
			y = y - (y>0)*((int)(y/twoPI))*twoPI - (y<0)*((int)(y/twoPI)-1)*twoPI;

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
			x = fmod(x, twoPI);
			x = (x < 0)*(twoPI+x) + (x > 0)*(x);
			y = fmod(y, twoPI);
			y = (y < 0)*(twoPI+y) + (y > 0)*(y);
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
	
	int In; 
	
	
	x = fmod(x, twoPI);
	x = (x < 0)*(twoPI+x) + (x > 0)*(x);
	y = fmod(y, twoPI);
	y = (y < 0)*(twoPI+y) + (y > 0)*(y);
	
	In = floor(y/twoPI * NY) * NX + floor(x/twoPI * NX); 
	//In = floor(x * NY) * NX + floor(y * NX); 
	
	return W_initial[In];

}








/*******************************************************************
*							   Zoom								   *
*******************************************************************/


void Zoom(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *Dev_ChiX_stack, double *Dev_ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *Dev_ChiX, double *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, double *W_real, cufftHandle cufftPlan_fine, double *W_initial, cufftDoubleComplex *Dev_Complex_fine, string simulationName, int simulation_num, double L)
{
	double *ws;
	ws = new double[Grid_fine->N];
	int save_ctr = 0;
	
	double xCenter = 0.54;
	double yCenter = 0.51;
	double width = 0.5;
	
	double xMin = xCenter - width/2;
	double xMax = xMin + width;
	double yMin = yCenter - width/2;
	double yMax = yMin + width;
	
	std::ostringstream ss;
	ss<<save_ctr;
	
	
	//save zooming effects
	for(int zoom_ctr = 0; zoom_ctr<10; zoom_ctr++){
		
		width *=  0.5;//0.99
		xMin = xCenter - width/2;
		xMax = xMin + width;
		yMin = yCenter - width/2;
		yMax = yMin + width;
		
		
		//kernel_apply_map_stack_to_W_custom<<<Gsf->blocksPerGrid, Gsf->threadsPerBlock>>>(devChiX_stack, devChiY_stack, devChiX, devChiY, devWs, stack_map_passed, Gc->NX, Gc->NY, Gc->h, Gsf->NX, Gsf->NY, Gsf->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial);	
		kernel_apply_map_stack_to_W_custom_part_All(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, W_real, Dev_Complex_fine, stack_length, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial, simulation_num);
		
		
		cudaMemcpy(ws, W_real, Grid_fine->sizeNReal, cudaMemcpyDeviceToHost);
		
		std::ostringstream ss2;
		ss2<<zoom_ctr;
		
		writeAllRealToBinaryFile(Grid_fine->N, ws, simulationName, "zoom_" + ss2.str());
	}
	
}


/*******************************************************************
*							    								   *
*******************************************************************/



















/******************************************************************/
/*******************************************************************
*							   Old								   *
*******************************************************************/
/******************************************************************/












////////////////////////////////////////////////////////////////////////
void test_fft_operations()
{
}

//void recompute_output_files(){}

////////////////////////////////////////////////////////////////////////
double compare_map_with_identity(double *chiX, double *chiY, int NX, int NY, double h)
{
return 0;
}

__global__ void kernel_compute_total_grid_Chi(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *gradChi, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
{
}

__global__ void kernel_compute_enstropy_increase_rate_factors(double *w, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, double ep)
{
}

__global__ void kernel_compute_enstropy_increase_rate_factors(double *wHsc, double *ChiX, double *ChiY, double *phi, double *div1, double *div2, int NXc, int NYc, double hc, int NXsc, int NYsc, double hsc, double ep)
{
}

////////////////////////////////////////////////////////////////////////
__global__ void kernel_advect_using_velocity_function(double *ChiX, double *ChiY, double *ChiDualX, double *ChiDualY,  int NXc, int NYc, double hc, double t, double dt, double ep)
{
}

////////////////////////////////////////////////////////////////////////
__global__ void kernel_apply_map_to_W(double *ChiX, double *ChiY, double *ws, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
{
}

__global__ void kernel_compare_map_stack_with_identity(double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *error, int stack_length, int NXc, int NYc, double hc, int NXs, int NYs, double hs)
{
}




























