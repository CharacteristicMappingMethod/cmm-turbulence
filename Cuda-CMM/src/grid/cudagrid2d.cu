#include "cudagrid2d.h"


TCudaGrid2D::TCudaGrid2D(int NX, int NY, double xRange)
{
	this->NX = NX;
	this->NY = NY;
	
	this->h = xRange/(float)NX;
	
	this->N = NX*NY;
	this->sizeNReal = sizeof(double)*N;
	this->sizeNComplex = sizeof(cufftDoubleComplex)*N;

	//block & grid
	threadsPerBlock.x = BLOCK_SIZE;
	threadsPerBlock.y = BLOCK_SIZE;
	threadsPerBlock.z = 1;

	blocksPerGrid.x = ceil((float)NX/threadsPerBlock.x);
	blocksPerGrid.y = ceil((float)NY/threadsPerBlock.y);
	blocksPerGrid.z = 1;

	// debug information about grid, maybe add a verbose parameter?
//	printf("Grid      : (%d, %d)\n", NX, NY);
//	printf("Block Dim : (%d, %d, %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
//	printf("Grid Dim  : (%d, %d, %d)\n\n", blocksPerGrid->x, blocksPerGrid->y, blocksPerGrid->z);
}


MapStack::MapStack(TCudaGrid2D *Grid, int cpu_map_num) {

	this->Grid = Grid;

	this->cpu_map_num = cpu_map_num;

	// set values
	map_stack_ctr = 0;
	Nb_array_RAM = 4;

	// initialize device map stack
	cudaMalloc((void **) &Dev_ChiX_stack, 4*Grid->sizeNReal);
	cudaMalloc((void **) &Dev_ChiY_stack, 4*Grid->sizeNReal);

	Host_ChiX_stack_RAM_0 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiY_stack_RAM_0 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiX_stack_RAM_1 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiY_stack_RAM_1 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiX_stack_RAM_2 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiY_stack_RAM_2 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiX_stack_RAM_3 = new double[cpu_map_num * 4*Grid->sizeNReal];
	Host_ChiY_stack_RAM_3 = new double[cpu_map_num * 4*Grid->sizeNReal];
}


// name says already everything actually
void MapStack::copy_map_to_host(double *Dev_ChiX, double *Dev_ChiY) {
	//saving map stack on device/host
	//cudaMemcpy(&Dev_ChiX_stack[map_stack_ctr*4*Grid->N], Dev_ChiX, 4*Grid->sizeNReal, cudaMemcpyDeviceToDevice);
	//cudaMemcpy(&Dev_ChiY_stack[map_stack_ctr*4*Grid->N], Dev_ChiY, 4*Grid->sizeNReal, cudaMemcpyDeviceToDevice);

    switch(map_stack_ctr / cpu_map_num){
        case 0:
            cudaMemcpy(&Host_ChiX_stack_RAM_0[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiX, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            cudaMemcpy(&Host_ChiY_stack_RAM_0[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiY, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            break;
		//cout<<"pos ram 0 : "<<map_stack_ctr%(cpu_map_num * map_stack_length)<<endl;
        case 1:
            cudaMemcpy(&Host_ChiX_stack_RAM_1[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiX, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            cudaMemcpy(&Host_ChiY_stack_RAM_1[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiY, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            break;
		//cout<<"pos ram 1 : "<<map_stack_ctr%(cpu_map_num * map_stack_length)<<endl;
        case 2:
            cudaMemcpy(&Host_ChiX_stack_RAM_2[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiX, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            cudaMemcpy(&Host_ChiY_stack_RAM_2[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiY, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            break;

        case 3:
            cudaMemcpy(&Host_ChiX_stack_RAM_3[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiX, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            cudaMemcpy(&Host_ChiY_stack_RAM_3[(map_stack_ctr%cpu_map_num)*4*Grid->N], Dev_ChiY, 4*Grid->sizeNReal, cudaMemcpyDeviceToHost);
            break;
    }
}

void MapStack::copy_map_to_device(int map_num) {

	switch (map_num / cpu_map_num) {
		case 0: {
			cudaMemcpy(Dev_ChiX_stack, &Host_ChiX_stack_RAM_0[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			cudaMemcpy(Dev_ChiY_stack, &Host_ChiY_stack_RAM_0[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			break;
		}
		case 1: {
			cudaMemcpy(Dev_ChiX_stack, &Host_ChiX_stack_RAM_1[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			cudaMemcpy(Dev_ChiY_stack, &Host_ChiY_stack_RAM_1[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			break;
		}
		case 2: {
			cudaMemcpy(Dev_ChiX_stack, &Host_ChiX_stack_RAM_2[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			cudaMemcpy(Dev_ChiY_stack, &Host_ChiY_stack_RAM_2[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			break;
		}
		case 3: {
			cudaMemcpy(Dev_ChiX_stack, &Host_ChiX_stack_RAM_3[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			cudaMemcpy(Dev_ChiY_stack, &Host_ChiY_stack_RAM_3[(map_num%cpu_map_num)*Grid->N*4], 4*Grid->sizeNReal, cudaMemcpyHostToDevice);
			break;
		}
	}
}

// free those resources
void MapStack::free_res() {
	delete [] Host_ChiX_stack_RAM_0;
	delete [] Host_ChiY_stack_RAM_0;
	delete [] Host_ChiX_stack_RAM_1;
	delete [] Host_ChiY_stack_RAM_1;
	delete [] Host_ChiX_stack_RAM_2;
	delete [] Host_ChiY_stack_RAM_2;
	delete [] Host_ChiX_stack_RAM_3;
	delete [] Host_ChiY_stack_RAM_3;
	cudaFree(Dev_ChiX_stack);
	cudaFree(Dev_ChiY_stack);
}



void get_max_min(TCudaGrid2D *G, double *var, double *min, double *max)
{
	//calculating max and min
	double var_min, var_max;
	
		for(int i=0; i<G->N; i++)
		{
			if(i==0)
			{
				var_min = var[i];
				var_max = var[i];
			}
			
			if(var_min > var[i])
				var_min = var[i];
				
			if(var_max < var[i])
				var_max = var[i];
		}
	
	*min = var_min;
	*max = var_max;
}


void Host_get_max_min(int len, double *Var_min, double *Var_max, double *min_f, double *max_f)
{
	//calculating max and min

	double min = Var_min[0];
	double max = Var_max[0];
	
	for(int i=0;i<len;i++)
	{			
		if(min > Var_min[i])
			min = Var_min[i];
			
		if(max < Var_max[i])
			max = Var_max[i];
	}

	*min_f = min;
	*max_f = max;
}


__global__ void Dev_get_max_min(int len, double *var, double *min, double *max)
{
    
	int In = threadIdx.x + blockDim.x * blockIdx.x;
	int Di = blockDim.x * gridDim.x;
	
	int pos = len / Di * In, step_pos = len / Di;
	//calculating max and min
	double var_min, var_max;

	var_min = var[pos];
	var_max = var[pos];
	
	for(int i=pos;i<pos + step_pos;i++)
	{
		if(var_min > var[i])
			var_min = var[i];
			
		if(var_max < var[i])
			var_max = var[i];
	}
	
	min[In] = var_min;
	max[In] = var_max;
}
