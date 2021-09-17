#include "cudasimulation2d.h"


////////////////////////////////////////////////////////////////////////
__global__ void kernel_init_diffeo(ptype *ChiX, ptype *ChiY, int NX, int NY, ptype h)
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


__global__ void kernel_update_map_from_dual(ptype *ChiX, ptype *ChiY, ptype *X, ptype *Y, int NX, int NY, ptype ep)																												
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	double c1 = 1.0/(4.0);
	double c2 = 1.0/(4.0*ep);
	double c3 = 1.0/(4.0*ep*ep);
	
	long int N = NX*NY;
	
	ChiX[    In] = ( (X[0*N+In] + X[1*N+In] + X[2*N+In] + X[3*N+In])*c1 );
	ChiY[    In] = ( (Y[0*N+In] + Y[1*N+In] + Y[2*N+In] + Y[3*N+In])*c1 );
		
	ChiX[1*N+In] = ( (X[0*N+In] + X[1*N+In] - X[2*N+In] - X[3*N+In])*c2 );
	ChiY[1*N+In] = ( (Y[0*N+In] + Y[1*N+In] - Y[2*N+In] - Y[3*N+In])*c2 );
		
	ChiX[2*N+In] = ( (X[0*N+In] - X[1*N+In] - X[2*N+In] + X[3*N+In])*c2 );
	ChiY[2*N+In] = ( (Y[0*N+In] - Y[1*N+In] - Y[2*N+In] + Y[3*N+In])*c2 );
		
	ChiX[3*N+In] = ( (X[0*N+In] - X[1*N+In] + X[2*N+In] - X[3*N+In])*c3 );
	ChiY[3*N+In] = ( (Y[0*N+In] - Y[1*N+In] + Y[2*N+In] - Y[3*N+In])*c3 );
}


__global__ void upsample(ptype *ChiX, ptype *ChiY, ptype *ChiX_2048, ptype *ChiY_2048, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)															// time cost
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;	
	
	//position
	ptype x = iX*hs;
	ptype y = iY*hs;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);		
	
	ChiX_2048[In] = x;
	ChiY_2048[In] = y;
}


__global__ void kernel_incompressibility_check(ptype *ChiX, ptype *ChiY, ptype *gradChi, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)															// time cost
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;	
	
	//position
	ptype x = iX*hs + 0.5*hs;
	ptype y = iY*hs + 0.5*hs;
	
	gradChi[In] = device_diffeo_grad(ChiX, ChiY, x, y, NXc, NYc, hc);
}


__global__ void kernel_advect_using_stream_hermite2(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *phi, ptype *phi_p, ptype *phi_p_p, int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep)			// time cost
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
    /*int NX = 512; //Dans l'interpolation remettre NXc Nyc hc
    int NY = 512;
    ptype h = twoPI/(float)NX;*/

    if(iX >= NXc || iY >= NYc)
		return;
	
	int In = iY*NXc + iX;
	
	long int N = NXc*NYc;
	
	//position
	ptype x = iX*hc;
	ptype y = iY*hc;
	
	//running through 4 neighbours (unrolled loops)
	ptype xep, yep;
	ptype phi_x, phi_y;
	
	
	ptype u, v, u_p, v_p, u_p_p, v_p_p;  // velocity at current and previous time steps
	ptype k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;  // different intermediate functions for RKThree
	ptype l[6] = {3.0,-3.0,1,1.875,-1.25,0.375};  // coefficients for lagrangian time interpolation of velocity
    ptype xf, yf;  // coordinates in loop
    
	ptype xf_p, yf_p;  // variables used for fixed-point iteration in ABTwo

    //ptype l[6] = {L1(t+dt, t, t-dt, t-2*dt), L2(t+dt, t, t-dt, t-2*dt), L3(t+dt, t, t-dt, t-2*dt), L1(t+dt/2, t, t-dt, t-2*dt), L2(t+dt/2, t, t-dt, t-2*dt), L3(t+dt/2, t, t-dt, t-2*dt)}; Work but slow

	// repeat for all 4 footpoints
    #pragma unroll 4
	FOR(k, 4)
	{
		// get position of footpoint
		xep = x+ep*(1 - 2*((k/2)%2 == 1));
		yep = y+ep*(1 - 2*(((k+1)/2)%2 == 1));
		
		xf = xep;
		yf = yep;

        if (TIME_INTEGRATION == "EulerExp") {
			device_hermite_interpolate_dx_dy(phi, xf, yf, &phi_x, &phi_y, NXc, NYc, hc);
			u_p =  phi_y;
			v_p = -phi_x;

			xf = xep - dt * u_p;
			yf = yep - dt * v_p;
        }

		// Adam-Bashfords of order two, where a fixed point iteration with 10 iterations is utilized
        else if (TIME_INTEGRATION == "ABTwo") {
			xf_p = xep; yf_p = yep;

			// fixed point iteration for xf,yf using previous foot points (self-correction)
            #pragma unroll 10
            FOR(ctr, 10)
            {
				//step 1
                device_hermite_interpolate_dx_dy(phi_p, xf_p, yf_p, &phi_x, &phi_y, NXc, NYc, hc);
                u_p =  phi_y; v_p = -phi_x;

                xf_p = xf - dt * u_p;
                yf_p = yf - dt * v_p;

				//step 2
                device_hermite_interpolate_dx_dy(phi_p, xf_p, yf_p, &phi_x, &phi_y, NXc, NYc, hc);
                u_p =  phi_y; v_p = -phi_x;

                device_hermite_interpolate_dx_dy(phi, xf , yf, &phi_x, &phi_y, NXc, NYc, hc);
                u =  phi_y; v = -phi_x;

                xf = xep - dt * (1.5*u - 0.5*u_p);
                yf = yep - dt * (1.5*v - 0.5*v_p);

            }
		}

		// RKThree time step utilizing some intermediate steps
        else if (TIME_INTEGRATION == "RKThree") {
			// compute u_tilde(X,t_n+1)
			device_hermite_interpolate_dx_dy(phi_p_p, xep, yep, &phi_x, &phi_y, NXc, NYc, hc);
			u_p_p = phi_y; v_p_p = -phi_x;

			device_hermite_interpolate_dx_dy(phi_p, xep, yep, &phi_x, &phi_y, NXc, NYc, hc);
			u_p = phi_y; v_p = -phi_x;

			device_hermite_interpolate_dx_dy(phi, xep, yep, &phi_x, &phi_y, NXc, NYc, hc);
			u = phi_y; v = -phi_x;

			// k1 = u_tilde(x,t_n+1)
			k1_x = l[0] * u + l[1] * u_p + l[2] * u_p_p;
			k1_y = l[0] * v + l[1] * v_p + l[2] * v_p_p;

			// compute u_tilde(x - dt*k1/2, t_n+1 - dt/2)
			device_hermite_interpolate_dx_dy(phi_p_p, xep - dt*k1_x/2, yep - dt*k1_y/2, &phi_x, &phi_y, NXc, NYc, hc);
			u_p_p = phi_y; v_p_p = -phi_x;

			device_hermite_interpolate_dx_dy(phi_p, xep - dt*k1_x/2, yep - dt*k1_y/2, &phi_x, &phi_y, NXc, NYc, hc);
			u_p = phi_y; v_p = -phi_x;

			device_hermite_interpolate_dx_dy(phi, xep - dt*k1_x/2, yep - dt*k1_y/2, &phi_x, &phi_y, NXc, NYc, hc);
			u = phi_y; v = -phi_x;

			//k2 = u_tilde(x - k1 dt/2, t_n+1 - dt/2)
			k2_x = l[3] * u + l[4]* u_p + l[5] * u_p_p;
			k2_y = l[3] * v + l[4] * v_p + l[5] * v_p_p;

			//compute u_tilde(x + dt * k1 - 2*dt*k2, t_n+1 - dt)
			device_hermite_interpolate_dx_dy(phi, xep + dt*k1_x - 2*dt*k2_x, yep + dt*k1_y - 2*dt*k2_y , &phi_x, &phi_y, NXc, NYc, hc);
			u = phi_y; v = -phi_x;

			// k3 = u_tilde(x = k1 dt - 2 k2 dt, t_n)
			k3_x = u;
			k3_y = v;

			xf = xep - dt * (k1_x + 4*k2_x + k3_x)/6;
			yf = yep - dt * (k1_y + 4*k2_y + k3_y)/6;
		}

        // scheme name not known, do euler explicit to avoid errors (a bit cheeky, i know, but i dont know how to throw errors yet)
        else {
			device_hermite_interpolate_dx_dy(phi, xf, yf, &phi_x, &phi_y, NXc, NYc, hc);
			u_p =  phi_y; v_p = -phi_x; xf = xep - dt * u_p; yf = yep - dt * v_p;
        }

		device_diffeo_interpolate(ChiX, ChiY, xf, yf, &ChiDualX[k*N+In], &ChiDualY[k*N+In], NXc, NYc, hc);
	}
}


/*******************************************************************
*						 Real and Complex						   *
*******************************************************************/
	
__global__ void kernel_real_to_complex(ptype *varR, cuPtype *varC, int NX, int NY)
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


__global__ void kernel_complex_to_real(ptype *varR, cuPtype *varC, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	varR[In] = varC[In].x;
}


__global__ void kernel_real_to_complex_H(ptype *varR, cuPtype *varC, int NX, int NY)
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


__global__ void kernel_complex_to_real_H(ptype *varR, cuPtype *varC, int NX, int NY)
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


__global__ void kernel_apply_map_stack_to_W(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ws, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype *W_initial)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	//position
	ptype x = iX*hs;
	ptype y = iY*hs;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);		
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);		
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x, y);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate(W_initial, x, y, NXs, NYs, hs);
		//ws[In] = device_initial_W_discret(x, y, W_initial, NXs, NYs);
	#endif
	
}


__global__ void kernel_apply_map_stack_to_W_custom(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ws, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype xl, ptype xr, ptype yl, ptype yr, ptype *W_initial)		// Zoom
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	ptype htemp = (xr - xl)/NXs;
	
	//position
	ptype x = xl + iX*htemp;
	ptype y = yl + iY*htemp;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);
		
	ws[In] = device_initial_W(x, y); //device_initial_W_discret(x, y)
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x, y);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate(W_initial, x, y, NXs, NYs, hs);
		//ws[In] = device_initial_W_discret(x, y, W_initial, NXs, NYs);
	#endif
	
}

/*
void kernel_apply_map_stack_to_W_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ChiX_stack_RAM, ptype *ChiY_stack_RAM, ptype *W_real, cuPtype *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype *W_initial)
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


void kernel_apply_map_stack_to_W_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *Host_ChiX_stack_RAM_0, ptype *Host_ChiY_stack_RAM_0, ptype *Host_ChiX_stack_RAM_1, ptype *Host_ChiY_stack_RAM_1, ptype *Host_ChiX_stack_RAM_2, ptype *Host_ChiY_stack_RAM_2, ptype *Host_ChiX_stack_RAM_3, ptype *Host_ChiY_stack_RAM_3, ptype *W_real, cuPtype *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype *W_initial)
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
	
	kernel_apply_map_stack_to_W_part_3<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
}


__global__ void kernel_apply_map_stack_to_W_part_1(ptype *ChiX, ptype *ChiY, cuPtype *x_y, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	//position
	ptype x = iX*hs;
	ptype y = iY*hs;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	
	x_y[In].x = x;
	x_y[In].y = y;
	
}

__global__ void kernel_apply_map_stack_to_W_part_2(ptype *ChiX_stack, ptype *ChiY_stack, cuPtype *x_y, int NXc, int NYc, ptype hc, int NXs, int NYs, int k)
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

__global__ void kernel_apply_map_stack_to_W_part_3(ptype *ws, cuPtype *x_y, int NXs, int NYs, ptype hs, ptype *W_initial)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	#ifndef DISCRET
		ws[In] = device_initial_W(x_y[In].x, x_y[In].y);
	#endif
	
	#ifdef DISCRET
		ws[In] = device_hermite_interpolate(W_initial, x_y[In].x, x_y[In].y, NXs, NYs, hs);
	#endif
	
}


void kernel_apply_map_stack_to_W_custom_part_All(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *Host_ChiX_stack_RAM_0, ptype *Host_ChiY_stack_RAM_0, ptype *Host_ChiX_stack_RAM_1, ptype *Host_ChiY_stack_RAM_1, ptype *Host_ChiX_stack_RAM_2, ptype *Host_ChiY_stack_RAM_2, ptype *Host_ChiX_stack_RAM_3, ptype *Host_ChiY_stack_RAM_3, ptype *W_real, cuPtype *Dev_Complex_fine, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype xl, ptype xr, ptype yl, ptype yr, ptype *W_initial)
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
	
	kernel_apply_map_stack_to_W_part_3<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
}


__global__ void kernel_apply_map_stack_to_W_custom_part_1(ptype *ChiX, ptype *ChiY, cuPtype *x_y, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, ptype xl, ptype xr, ptype yl, ptype yr)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	
	ptype htemp = (xr - xl)/NXs;
	
	//position
	ptype x = xl + iX*htemp;
	ptype y = yl + iY*htemp;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);
	
	x_y[In].x = x;
	x_y[In].y = y;
	
}


__global__ void cut_off_scale(cuPtype *W, int NX)
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
	if (i >= NX/2 && j <= NX/2)
		if (((NX-i)*(NX-i) + j*j) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	if (i <= NX/2 && j >= NX/2)
		if ((i*i + (NX-j)*(NX-j)) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	if (i >= NX/2 && j >= NX/2)
		if (((NX-i)*(NX-i) + (NX-j)*(NX-j)) > NX*NX/9){
			W[In].x = 0;
			W[In].y = 0;
		}
	
	if (In == 0){
		W[In].x = 0;
		W[In].y = 0;
	}
	
}


/*******************************************************************
*						 										   *
*******************************************************************/


__global__ void kernel_compare_vorticity_with_initial(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *error, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;
	long int N = NXc*NYc;	
	
	//position
	ptype x = iX*hs;
	ptype y = iY*hs;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x, &y, NXc, NYc, hc);		
	for(int k = stack_length - 1; k >= 0; k--)
		device_diffeo_interpolate(&ChiX_stack[k*N*4], &ChiY_stack[k*N*4], x, y, &x, &y, NXc, NYc, hc);		
	
	error[In] = fabs(device_initial_W(x, y) - device_initial_W(iX*hs, iY*hs));
}




__global__ void kernel_apply_map_and_sample_from_hermite(ptype *ChiX, ptype *ChiY, ptype *fs, ptype *H, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs, int NXh, int NYh, ptype hh)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
    /*int NX = 128; //Dans l'interpolation remettre NXc Nyc hc
    int NY = 128;
    ptype h = twoPI/(float)NX;*/

	if(iX >= NXs || iY >= NYs)
		return;
	
	int In = iY*NXs + iX;	
	
	//position
	ptype x = iX*hs;
	ptype y = iY*hs;
	
	ptype x2, y2;
	
	device_diffeo_interpolate(ChiX, ChiY, x, y, &x2, &y2, NXc, NYc, hc);
	
	
	fs[In] = device_hermite_interpolate(H, x2, y2, NXh, NYh, hh);
	
}


/*******************************************************************
*						 										   *
*******************************************************************/

////////////////////////////////////////////////////////////////////////
__global__ void kernel_sample_on_coarse_grid(cuPtype *AcOut, cuPtype *AfOut, int NXc, int NYc, ptype hc, int NXf, int NYf, ptype hf)
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


__global__ void kernel_normalize(cuPtype *F, int NX, int NY)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	ptype N = NX*NY;
	
	F[In].x /= (ptype)N;
	F[In].y /= (ptype)N;
}


/*******************************************************************
*						 Initial condition						   *
*******************************************************************/

__device__ ptype device_initial_W(ptype x, ptype y)
{
// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"

	if(PROBLEM_CODE == "4_nodes")
	{
		x = x - (x>0)*((int)(x/twoPI))*twoPI - (x<0)*((int)(x/twoPI)-1)*twoPI;
		y = y - (y>0)*((int)(y/twoPI))*twoPI - (y<0)*((int)(y/twoPI)-1)*twoPI;
		
		return cos(x) + cos(y) + 0.6*cos(2*x) + 0.2*cos(3*x);
	}
	else if(PROBLEM_CODE == "quadropole")
	{
		ptype ret = 0;
		for(int iy = -2; iy <= 2; iy++)
			for(int ix = -2; ix <= 2; ix++)
			{
				ptype dx = x - PI/2 + ix * 2*PI; 
				ptype dy = y - PI/2 + iy * 2*PI;
				ptype A = 0.6258473;
				ptype s = 0.5;
				ptype B = A/(s*s*s*s) * (dx * dy) * (dx*dx + dy*dy - 6*s*s);
				ptype D = (dx*dx + dy*dy)/(2*s*s);
				ret += B * exp(-D);
			}
			return ret;
	}
	else if(PROBLEM_CODE == "two_votices")
	{
		ptype ret = 0;
		for(int iy = -1; iy <= 1; iy++)
			for(int ix = -1; ix <= 1; ix++)
			{
				ret += sin(0.5*(x + twoPI*ix))*sin(0.5*(x + twoPI*ix))*sin(0.5*((y + twoPI*iy) + twoPI*iy))*sin(0.5*((y + twoPI*iy) + twoPI*iy)) * (exp(-(((x + twoPI*ix) - PI)*((x + twoPI*ix) - PI) + ((y + twoPI*iy) - 0.33*twoPI)*((y + twoPI*iy) - 0.33*twoPI))*5) + 
											exp(-(((x + twoPI*ix) - PI)*((x + twoPI*ix) - PI) + ((y + twoPI*iy) - 0.67*twoPI)*((y + twoPI*iy) - 0.67*twoPI))*5));		 //two votices of same size
			}
		return ret;
	}
	else if(PROBLEM_CODE == "three_vortices")
	{
		//three vortices
		ptype ret = 0;
		ptype LX = PI/2;
		ptype LY = PI/(2.0*sqrt(2.0));
		
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
	}
	else if(PROBLEM_CODE == "single_shear_layer")
	{
		//single shear layer
		ptype delta = 50;
		ptype delta2 = 0.01;
		ptype ret = 0;
		for(int iy = -1; iy <= 1; iy++)
			for(int iy = -1; iy <= 1; iy++)
				{
					ret +=    (1 + delta2 * cos(2*x))  *    exp( - delta * (y - PI) * (y - PI) ); 
				}
		ret /= 9;
		return ret;
	}
	else if(PROBLEM_CODE == "turbulence_gaussienne")
	{	
		x = fmod(x, twoPI);
		x = (x < 0)*(twoPI+x) + (x > 0)*(x);
		y = fmod(y, twoPI);
		y = (y < 0)*(twoPI+y) + (y > 0)*(y);
		int NB_gaus = 8;		//NB_gaus = 6;sigma = 0.24;
		ptype sigma = 0.2;
		ptype ret = 0;
		for(int mu_x = 0; mu_x < NB_gaus; mu_x++){
			for(int mu_y = 0; mu_y < NB_gaus; mu_y++){
				ret += 1/(twoPI*sigma*sigma)*exp(-((x-mu_x*twoPI/(NB_gaus-1))*(x-mu_x*twoPI/(NB_gaus-1))/(2*sigma*sigma)+(y-mu_y*twoPI/(NB_gaus-1))*(y-mu_y*twoPI/(NB_gaus-1))/(2*sigma*sigma))); 
			}
		}
		for(int mu_x = 0; mu_x < NB_gaus-1; mu_x++){
			for(int mu_y = 0; mu_y < NB_gaus-1; mu_y++){
				curandState_t state_x;
				curand_init((mu_x+1)*mu_y*mu_y, 0, 0, &state_x);
				ptype RAND_gaus_x = ((ptype)(curand(&state_x)%1000)-500)/100000;
				curandState_t state_y;
				curand_init((mu_y+1)*mu_x*mu_x, 0, 0, &state_y);
				ptype RAND_gaus_y = ((ptype)(curand(&state_y)%1000)-500)/100000;
				ret -= 1/(twoPI*sigma*sigma)*exp(-((x-(2*mu_x+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_x)*(x-(2*mu_x+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_x)/(2*sigma*sigma)+(y-(2*mu_y+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_y)*(y-(2*mu_y+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_y)/(2*sigma*sigma))); 
			}
		}
		//curandState_t state;
		//curand_init(floor(y * 16384) * 16384 + floor(x * 16384), 0, 0, &state);
		//ret *= 1+((ptype)(curand(&state)%1000)-500)/100000;
		return ret - 0.008857380480028442;
	}
	else	//default case goes to stationary
	{
		x = x - (x>0)*((int)(x/twoPI))*twoPI - (x<0)*((int)(x/twoPI)-1)*twoPI;
		y = y - (y>0)*((int)(y/twoPI))*twoPI - (y<0)*((int)(y/twoPI)-1)*twoPI;
		
		return cos(x)*cos(y);
	}

}



__device__ ptype device_initial_W_discret(ptype x, ptype y, ptype *W_initial, int NX, int NY){
	
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


void Zoom(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX_stack, ptype *Dev_ChiY_stack, ptype *Host_ChiX_stack_RAM_0, ptype *Host_ChiY_stack_RAM_0, ptype *Host_ChiX_stack_RAM_1, ptype *Host_ChiY_stack_RAM_1, ptype *Host_ChiX_stack_RAM_2, ptype *Host_ChiY_stack_RAM_2, ptype *Host_ChiX_stack_RAM_3, ptype *Host_ChiY_stack_RAM_3, ptype *Dev_ChiX, ptype *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, ptype *W_real, cufftHandle cufftPlan_fine, ptype *W_initial, cuPtype *Dev_Complex_fine, string simulationName, ptype L)
{
	ptype *ws;
	ws = new ptype[Grid_fine->N];
	int save_ctr = 0;
	
	ptype xCenter = 0.54; 
	ptype yCenter = 0.51; 
	ptype width = 0.5;
	
	ptype xMin = xCenter - width/2;
	ptype xMax = xMin + width;
	ptype yMin = yCenter - width/2;
	ptype yMax = yMin + width;
	
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
		kernel_apply_map_stack_to_W_custom_part_All(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, W_real, Dev_Complex_fine, stack_length, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial);
		
		
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
ptype compare_map_with_identity(ptype *chiX, ptype *chiY, int NX, int NY, ptype h)
{
return 0;
}

__global__ void kernel_compute_total_grid_Chi(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *gradChi, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)
{
}

__global__ void kernel_compute_enstropy_increase_rate_factors(ptype *w, ptype *phi, ptype *div1, ptype *div2, int NXc, int NYc, ptype hc, ptype ep)
{
}

__global__ void kernel_compute_enstropy_increase_rate_factors(ptype *wHsc, ptype *ChiX, ptype *ChiY, ptype *phi, ptype *div1, ptype *div2, int NXc, int NYc, ptype hc, int NXsc, int NYsc, ptype hsc, ptype ep)
{
}

////////////////////////////////////////////////////////////////////////
__global__ void kernel_advect_using_velocity_function(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY,  int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep)
{
}

__global__ void kernel_advect_using_stream_hermite(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *phi, int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep)
{
}

////////////////////////////////////////////////////////////////////////
__global__ void kernel_apply_map_to_W(ptype *ChiX, ptype *ChiY, ptype *ws, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)
{
}

__global__ void kernel_compare_map_stack_with_identity(ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *error, int stack_length, int NXc, int NYc, ptype hc, int NXs, int NYs, ptype hs)
{
}




























