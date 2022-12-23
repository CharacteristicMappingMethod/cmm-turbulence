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

#include "cmm-grid2d.h"


TCudaGrid2D::TCudaGrid2D (int NX, int NY, int NZ, double *bounds)
{
	this->NX = NX;
	this->NX_fft = (int)(NX/2.0+1);
	this->NY = NY;
	this->NZ = NZ;

	this->h = (bounds[1] - bounds[0]) / (float)NX;  // for quadratic problems, is used everywhere so changing it is tedious

	this->hx = (bounds[1] - bounds[0]) / (float)NX;
	this->hy = (bounds[3] - bounds[2]) / (float)NY;
	this->hz = (bounds[5] - bounds[4]) / (float)NZ;

	for (int i_b = 0; i_b < 4; ++i_b) {
		this->bounds[i_b] = bounds[i_b];
	}

	this->N = NX*NY*NZ;
	this->Nfft = NX_fft*NY*NZ;

	this->sizeNReal = sizeof(double)*N;
	this->sizeNComplex = sizeof(cufftDoubleComplex)*N;
	this->sizeNfft = sizeof(cufftDoubleComplex)*Nfft;  // fft D2Z and Z2D size

	//block & grid
	threadsPerBlock = {BLOCK_SIZE, BLOCK_SIZE, 1};

	blocksPerGrid.x = ceil(NX/(double)threadsPerBlock.x);
	blocksPerGrid.y = ceil(NY/(double)threadsPerBlock.y);
	blocksPerGrid.z = ceil(NZ/(double)threadsPerBlock.z);

	fft_blocks.x = ceil((NX+1)/2.0/(double)threadsPerBlock.x);
	fft_blocks.y = ceil(NY/(double)threadsPerBlock.y);
	fft_blocks.z = ceil(NZ/(double)threadsPerBlock.z);
}


void fill_grid(TCudaGrid2D *Grid, int NX, int NY, int NZ, double *bounds) {
	Grid->NX = NX;
	Grid->NX_fft = (int)(NX/2.0+1);
	Grid->NY = NY;
	Grid->NZ = NZ;

	Grid->h = (bounds[1] - bounds[0]) / (float)NX;  // for quadratic problems, is used everywhere so changing it is tedious

	Grid->hx = (bounds[1] - bounds[0]) / (float)NX;
	Grid->hy = (bounds[3] - bounds[2]) / (float)NY;
	Grid->hz = (bounds[5] - bounds[4]) / (float)NZ;

	for (int i_b = 0; i_b < 6; ++i_b) {
		Grid->bounds[i_b] = bounds[i_b];
	}

	Grid->N = NX*NY*NZ;
	Grid->Nfft = Grid->NX_fft*NY*NZ;

	Grid->sizeNReal = sizeof(double)*Grid->N;
	Grid->sizeNComplex = sizeof(cufftDoubleComplex)*Grid->N;
	Grid->sizeNfft = sizeof(cufftDoubleComplex)*Grid->Nfft;  // fft D2Z and Z2D size

	//block & grid
	Grid->threadsPerBlock = {BLOCK_SIZE, BLOCK_SIZE, 1};

	Grid->blocksPerGrid.x = ceil(NX/(double)Grid->threadsPerBlock.x);
	Grid->blocksPerGrid.y = ceil(NY/(double)Grid->threadsPerBlock.y);
	Grid->blocksPerGrid.z = ceil(NZ/(double)Grid->threadsPerBlock.z);

	Grid->fft_blocks.x = ceil((NX+1)/2.0/(double)Grid->threadsPerBlock.x);
	Grid->fft_blocks.y = ceil(NY/(double)Grid->threadsPerBlock.y);
	Grid->fft_blocks.z = ceil(NZ/(double)Grid->threadsPerBlock.z);
}


MapStack::MapStack(TCudaGrid2D *Grid, int cpu_map_num)
{

	this->Grid = Grid;

	this->cpu_map_num = cpu_map_num;

	// pre-set values
	map_stack_ctr = 0;

	// initialize device map with host map stack
	cudaMalloc((void **) &Dev_ChiX_stack, 4*Grid->sizeNReal);
	cudaMalloc((void **) &Dev_ChiY_stack, 4*Grid->sizeNReal);

	// host pointer array pointing to each individual map
	Host_ChiX_stack_RAM = new double*[cpu_map_num];
	Host_ChiY_stack_RAM = new double*[cpu_map_num];

	// initialize memory for each individual map, separate so non-contigous in memory
	for (int i_m = 0; i_m < cpu_map_num; i_m++) {
		Host_ChiX_stack_RAM[i_m] = new double[4*Grid->sizeNReal];
		Host_ChiY_stack_RAM[i_m] = new double[4*Grid->sizeNReal];
	}
}


// copy inserted map into last position at map stack, after 1/4 of all total maps, start using new array
void MapStack::copy_map_to_host(double *Dev_ChiX, double *Dev_ChiY) {
	cudaMemcpy(Host_ChiX_stack_RAM[map_stack_ctr], Dev_ChiX, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
	cudaMemcpy(Host_ChiY_stack_RAM[map_stack_ctr], Dev_ChiY, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
    map_stack_ctr++;
}
// copy map to device, map_num decides which map to take out
void MapStack::copy_map_to_device(int map_num) {
	cudaMemcpy(Dev_ChiX_stack, Host_ChiX_stack_RAM[map_num], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiY_stack, Host_ChiY_stack_RAM[map_num], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
}

// free those resources
void MapStack::free_res() {
	// delete memory for each individual map
	for (int i_m = 0; i_m < cpu_map_num; i_m++) {
		delete [] Host_ChiX_stack_RAM[i_m];
		delete [] Host_ChiY_stack_RAM[i_m];
	}
	delete [] Host_ChiX_stack_RAM;
	delete [] Host_ChiY_stack_RAM;
	cudaFree(Dev_ChiX_stack);
	cudaFree(Dev_ChiY_stack);
}
