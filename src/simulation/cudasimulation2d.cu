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


__global__ void kernel_advect_using_stream_hermite2(ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *phi, ptype *phi_p, int NXc, int NYc, ptype hc, ptype t, ptype dt, ptype ep)			// time cost	
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
	
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
	
	
	ptype u, v, u_p, v_p;
	ptype xf, yf, xf_p, yf_p;

	
	#pragma unroll 4
	FOR(k, 4)
	{
		
		xep = x+ep*(1 - 2*((k/2)%2 == 1));
		yep = y+ep*(1 - 2*(((k+1)/2)%2 == 1));
		
		xf = xf_p = xep;
		yf = yf_p = yep;
		
		
		//#pragma unroll 10
		//FOR(ctr, 10)
		//{
		
		//adam-B 2
		
		//step 1
		device_hermite_interpolate_dx_dy(phi_p, xf_p, yf_p, &phi_x, &phi_y, NXc, NYc, hc);
		u_p =  phi_y;
		v_p = -phi_x;
		
		xf_p = xf - dt * u_p;
		yf_p = yf - dt * v_p;
	
		//step 2
		device_hermite_interpolate_dx_dy(phi_p, xf_p, yf_p, &phi_x, &phi_y, NXc, NYc, hc);
		
		u_p =  phi_y;
		v_p = -phi_x;
		
		device_hermite_interpolate_dx_dy(phi, xf, yf, &phi_x, &phi_y, NXc, NYc, hc);
		
		u =  phi_y;
		v = -phi_x;
		
		xf = xep - dt * (1.5*u - 0.5*u_p);	
		yf = yep - dt * (1.5*v - 0.5*v_p);	
		
		//}
		
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




void Zoom(string simulationName, ptype L, TCudaGrid2D *Gc, TCudaGrid2D *Gsf, ptype *devChiX_stack, ptype *devChiY_stack, ptype *devChiX, ptype *devChiY, ptype *devWs, int stack_map_passed, ptype *W_initial)
{
	ptype *ws;
	ws = new ptype[Gsf->N];
	int save_ctr = 0;
	
	ptype xCenter = 0.58; 
	ptype yCenter = 0.42; 
	ptype width = 0.5;
	
	ptype xMin = xCenter - width/2;
	ptype xMax = xMin + width;
	ptype yMin = yCenter - width/2;
	ptype yMax = yMin + width;
	
	std::ostringstream ss;
	ss<<save_ctr;
	
	
	//save zooming effects
	for(int zoom_ctr = 0; zoom_ctr<5; zoom_ctr++){
		
		width *=  0.75;
		xMin = xCenter - width/2;
		xMax = xMin + width;
		yMin = yCenter - width/2;
		yMax = yMin + width;
		
		
		kernel_apply_map_stack_to_W_custom<<<Gsf->blocksPerGrid, Gsf->threadsPerBlock>>>(devChiX_stack, devChiY_stack, devChiX, devChiY, devWs, stack_map_passed, Gc->NX, Gc->NY, Gc->h, Gsf->NX, Gsf->NY, Gsf->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial);	
		
		
		cudaMemcpy(ws, devWs, Gsf->sizeNReal, cudaMemcpyDeviceToHost);
		
		std::ostringstream ss2;
		ss2<<zoom_ctr;
		
		writeAllRealToBinaryFile(Gsf->N, ws, simulationName, "zoom_" + ss2.str());
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




























