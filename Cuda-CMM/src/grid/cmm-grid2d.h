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

#ifndef __CUDA_GRID_2D_H__
#define __CUDA_GRID_2D_H__

#include <cufft.h>
#include <cufftXt.h>

#define PI 		3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
#define twoPI 	6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136

//the grid is assumed to span the space [0, NX*h] X [0, NY*h]
struct TCudaGrid2D
{
	public:
		int NX, NY, NZ, NX_fft;
		long long int N, Nfft;
		double h;

		double hx, hy, hz;
		double bounds[6];

		dim3 threadsPerBlock, blocksPerGrid, fft_blocks;

		long long int sizeNReal, sizeNComplex, sizeNfft;

		TCudaGrid2D(int NX, int NY, int NZ, double *bounds);
};

void fill_grid(TCudaGrid2D *Grid, int NX, int NY, int NZ, double *bounds);

// class for map stack thingies, because I am always just transferring this everywhere
class MapStack {
public:
	double *Host_ChiX_stack_RAM_0, *Host_ChiY_stack_RAM_0,
		   *Host_ChiX_stack_RAM_1, *Host_ChiY_stack_RAM_1,
		   *Host_ChiX_stack_RAM_2, *Host_ChiY_stack_RAM_2,
		   *Host_ChiX_stack_RAM_3, *Host_ChiY_stack_RAM_3,
		   *Dev_ChiX_stack, *Dev_ChiY_stack;
	int cpu_map_num, Nb_array_RAM;
	int map_stack_ctr;

	TCudaGrid2D *Grid;

	MapStack(TCudaGrid2D *Grid, int cpu_map_num);

	void copy_map_to_host(double *Dev_ChiX, double *Dev_ChiY);
	void copy_map_to_device(int map_num);

	void free_res();
};



#ifdef __CUDACC__

	#define BLOCK_SIZE 16	// most devices have maximum thread size of 1024, i don't know why this is 16*16=256

	__global__ void Dev_get_max_min(int len, double *var, double *min, double *max);
#endif

#endif


