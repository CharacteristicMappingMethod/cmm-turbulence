#include "cudagrid2d.h"


Logger::Logger(string simulationName)
{
	fileName = "data/" + simulationName + "/log.txt";
	file.open(fileName.c_str(), ios::out);
	
	if(!file)
	{
		cout<<"Unable to open log file.. exitting\n";
		exit(0);
	}
	else
	{
		file<<simulationName<<endl;
		file.close();
	}
}


void Logger::push(string message)
{
	file.open(fileName.c_str(), ios::out | ios::app);
	
	if(file)
	{
		file<<"["<<currentDateTime()<<"]\t";
		file<<message<<endl;
		file.close(); 
	}
}


void Logger::push()
{
	push(buffer);
}


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
//	printf("Grid Dim  : (%d, %d, %d)\n\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
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


const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

