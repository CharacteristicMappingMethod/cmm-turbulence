/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/CharacteristicMappingMethod/cmm-turbulence
*
******************************************************************************************************************************/

#include "cmm-grid2d.h"

// debugging, using printf
#include "stdio.h"
#include <math.h>

TCudaGrid2D::TCudaGrid2D (int NX, int NY, int NZ, double *bounds)
{
	this->NX = NX;
	this->NX_fft = (int)(NX/2.0+1);
	this->NY = NY;
	this->NZ = NZ;

	this->hx = (bounds[1] - bounds[0]) / (float)NX;
	this->hy = (bounds[3] - bounds[2]) / (float)NY;
	this->hz = (bounds[5] - bounds[4]) / (float)NZ;

	for (int i_b = 0; i_b < 6; ++i_b) {
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


CmmVar2D::CmmVar2D(int NX, int NY, int NZ, double *bounds, int type) {
	// fill variables for grid
	this->Grid = (TCudaGrid2D *)malloc(sizeof(TCudaGrid2D));
	fill_grid(this->Grid, NX, NY, NZ, bounds);
	this->type = type;

	// normal array
	if (type == 0) {
		cudaMalloc((void**)&Dev_var, Grid->sizeNReal);
		RAM_size = Grid->sizeNReal / 1e6;
	}
	// normal Hermite array
	else if (type == -1) {
		cudaMalloc((void**)&Dev_var, 4*Grid->sizeNReal);
		RAM_size = 4*Grid->sizeNReal / 1e6;
	}
	// Hermite array with fft extension
	else if (type == -2) {
		cudaMalloc((void**)&Dev_var, 3*Grid->sizeNReal + Grid->sizeNfft);
		Dev_var += 2*Grid->Nfft - Grid->N; // shift to hide beginning buffer
		RAM_size = (3*Grid->sizeNReal + Grid->sizeNfft) / 1e6;
	}
	// Empty array, useful if only plan and grids are needed or if option is disabled
	else if (type == -10) {
		cudaMalloc((void**)&Dev_var, sizeof(double));
		RAM_size = (sizeof(double)) / 1e6;
	}
	// assume range 1-5 to be lagrange orders for psi, this is pretty dirty but then I dont have to pass an additional array
	else if (type > 0 && type <= 5){
		cudaMalloc((void**)&Dev_var, 3*Grid->sizeNReal+Grid->sizeNfft + (type-1)*4*Grid->sizeNReal);
		Dev_var += 2*Grid->Nfft - Grid->N; // shift to hide beginning buffer
		RAM_size = (3*Grid->sizeNReal+Grid->sizeNfft + (type-1)*4*Grid->sizeNReal) / 1e6;
	}

	cufftCreate(&plan_D2Z); cufftCreate(&plan_Z2D);
	cufftCreate(&plan_1D_D2Z); cufftCreate(&plan_1D_Z2D);
	// disable automatic setting of workareas, so that they all later share the same work area
	cufftSetAutoAllocation(plan_D2Z, 0); cufftSetAutoAllocation(plan_Z2D, 0);
	cufftSetAutoAllocation(plan_1D_D2Z, 0); cufftSetAutoAllocation(plan_1D_Z2D, 0);
	cufftMakePlan2d(plan_D2Z, Grid->NX, Grid->NY, CUFFT_D2Z, &plan_size[0]);
	cufftMakePlan2d(plan_Z2D, Grid->NX, Grid->NY, CUFFT_Z2D, &plan_size[1]);
	cufftMakePlan1d(plan_1D_D2Z, Grid->NX, CUFFT_D2Z, 1, &plan_1D_size[0]);
	cufftMakePlan1d(plan_1D_Z2D, Grid->NX, CUFFT_Z2D, 1, &plan_1D_size[1]);

	// ToDo Asserting
}

// Set the work area of the cufft_plans, since no parallel fft is performed, all plans share the same work area
void CmmVar2D::setWorkArea(void *fft_work_area) {
	cufftSetWorkArea(plan_D2Z, fft_work_area); cufftSetWorkArea(plan_Z2D, fft_work_area);
	cufftSetWorkArea(plan_1D_D2Z, fft_work_area); cufftSetWorkArea(plan_1D_Z2D, fft_work_area);
}

// free resources
void CmmVar2D::free_res() {
	// free variable, take reshift into account
	if (type == -2 || (type > 0 && type <= 5)) cudaFree(Dev_var - 2*Grid->Nfft + Grid->N);  // Dev_var - 2*Grid->Nfft + Grid->N
	else cudaFree(Dev_var);
	cufftDestroy(plan_D2Z); cufftDestroy(plan_Z2D);
}

CmmPart::CmmPart(size_t num, double tau_p) {
	this->num = num;
	this->sizeN = 2*num*sizeof(double);
	this->tau_p = tau_p;
	this->block = ceil(num / (double)this->thread);

	cudaMalloc((void**)&Dev_var, 2*num*sizeof(double));
	RAM_size = this->sizeN / 1e6;
}
void CmmPart::free_res() {
	cudaFree(Dev_var);
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


// Define the __device__ function to calculate velocity coordinate
__device__ double calculate_velocity_coordinate(TCudaGrid2D Grid, int iY) {
	// This method capsulates the calculation of the velocity coordinate
    // return Grid.bounds[2] + iY * Grid.hy + Grid.hy / 2.0;
	return Grid.bounds[2] + iY * Grid.hy;
}

