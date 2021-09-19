#include "cudagrid2d.h"



ptype ep = 1e-6;
ptype epdx[4] = {ep, ep, -ep, -ep};
ptype epdy[4] = {ep, -ep, -ep, ep};




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


TCudaGrid2D::TCudaGrid2D(int NX, int NY, ptype xRange)
{
	this->NX = NX;
	this->NY = NY;
	
	this->h = xRange/(float)NX;
	
	N = NX*NY;
	this->sizeNReal = sizeof(ptype)*N;
	this->sizeNComplex = sizeof(cuPtype)*N;	


			//block & grid
			threadsPerBlock.x = BLOCK_SIZE;
			threadsPerBlock.y = BLOCK_SIZE;
			threadsPerBlock.z = 1;
			
			blocksPerGrid.x = ceil((float)NX/threadsPerBlock.x);
			blocksPerGrid.y = ceil((float)NY/threadsPerBlock.y);
			blocksPerGrid.z = 1;
			
			
			printf("Grid      : (%d, %d)\n", NX, NY);
			printf("Block Dim : (%d, %d, %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
			printf("Grid Dim  : (%d, %d, %d)\n\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
			
}


void get_max_min(TCudaGrid2D *G, ptype *var, ptype *min, ptype *max)
{
	//calculating max and min
	ptype var_min, var_max;
	
		FOR(i, G->N)
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


void Host_get_max_min(int len, ptype *Var_min, ptype *Var_max, ptype min, ptype max)
{
	//calculating max and min

	min = Var_min[0];
	max = Var_max[0];
	
	for(int i=0;i<len;i++)
	{			
		if(min > Var_min[i])
			min = Var_min[i];
			
		if(max < Var_max[i])
			max = Var_max[i];
	}
	//printf("min max : %f, %f \n", min, max);
}


__global__ void Dev_get_max_min(int len, ptype *var, ptype *min, ptype *max)
{
    
	int In = threadIdx.x + blockDim.x * blockIdx.x;
	int Di = blockDim.x * gridDim.x;
	
	int pos = len / Di * In, step_pos = len / Di;
	//calculating max and min
	ptype var_min, var_max;

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



/*******************************************************************
*						  Fourier operations					   *
*******************************************************************/


__global__ void kernel_fft_lap(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h)						// Laplace operator in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
		
	if(iX >= NX || iY >= NY)
		return;
		
	if(iX == 0 && iY == 0)
	{
		BOut[0].x = BOut[0].y = 0;
		return;
	}
	
	int In = iY*NX + iX;	
	
	ptype kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);						// twoPI/(h*NX) = 1 
	ptype ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);
	ptype k2 = kx*kx + ky*ky;
	
	ptype x = -AOut[In].x * k2;
	ptype y = -AOut[In].y * k2;
	
	BOut[In].x = x;	
	BOut[In].y = y;	
}

__global__ void kernel_fft_iLap(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h)						// Inverse laplace operator in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
		
	if(iX >= NX || iY >= NY)
		return;
		
	if(iX == 0 && iY == 0)
	{
		BOut[0].x = BOut[0].y = 0;
		return;
	}
	
	int In = iY*NX + iX;	
	
	ptype kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);						// twoPI/(h*NX) = twoPI/(twoPI/NX*NX) = 1 
	ptype ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);
	ptype k2 = kx*kx + ky*ky;
	
	ptype x = -AOut[In].x / k2;
	ptype y = -AOut[In].y / k2;
	
	BOut[In].x = x;	
	BOut[In].y = y;	
}

__global__ void kernel_fft_dx(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h)						// x derivative in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
		
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	ptype kx = twoPI/(h*NX) * (iX - (iX>NX/2)*NX);
	
	ptype x = -AOut[In].y * kx;
	ptype y =  AOut[In].x * kx;
	
	BOut[In].x = x;	
	BOut[In].y = y;	
}

__global__ void kernel_fft_dy(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h)						// y derivative in Fourier space
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x);
	int iY = (blockDim.y * blockIdx.y + threadIdx.y);
		
	if(iX >= NX || iY >= NY)
		return;
	
	int In = iY*NX + iX;	
	
	ptype ky = twoPI/(h*NY) * (iY - (iY>NY/2)*NY);
	
	ptype x = -AOut[In].y * ky;
	ptype y =  AOut[In].x * ky;
	
	BOut[In].x = x;	
	BOut[In].y = y;	
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


/*******************************************************************
*						  Writting data							   *
*******************************************************************/


void writeRealToFile(TCudaGrid2D *G, ptype *var, string fileName)
{
	fileName = "data/" + fileName + ".csv";
	ofstream file(fileName.c_str(), ios::out);
	
		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
			return;
		}
	
	char buffer[50];
	int n=0;
	FOR(j, G->NY)
	{
		FOR(i, G->NX)
		{
			sprintf(buffer, "%e", var[n]);
			file<<buffer<< ",";
			n++;
		}
		file << "\n";
	}
	file.close();
}


void writeRealToBinaryFile(TCudaGrid2D *G, ptype *var, string fileName)
{
	fileName = "data/" + fileName + ".data";
	ofstream file(fileName.c_str(), ios::out | ios::binary);
	
		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
			return;
		}
	
	int n=0;
	FOR(j, G->NY)
	{
		FOR(i, G->NX)
		{
			file.write( (char*) &var[n], sizeof(ptype) );
			n++;
		}
	}
	file.close();
}


int readRealFromBinaryFile(TCudaGrid2D *G, ptype *var, string fileName)
{
	fileName = "data/" + fileName + ".data";
	ifstream file(fileName.c_str(), ios::in | ios::binary);
	
		if(!file)
		{
			//cout<<"Error saving file. Unable to open : "<<fileName<<endl;
			return -1;
		}
	
	int n=0;
	FOR(j, G->NY)
	{
		FOR(i, G->NX)
		{
			file.read( (char*) &var[n], sizeof(ptype) );
			n++;
		}
	}
	file.close();
	
	return 0;
}


void writeDiffeoToBinaryFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr)
{
	std::ostringstream ss;
	ss<<ctr;
	fileName = simulationName + "/all_map_data/" + fileName + ss.str();
	
	writeRealToBinaryFile(G,  ChiX, 			fileName + "_X_F");
	writeRealToBinaryFile(G, &ChiX[1*G->N], 	fileName + "_X_Fx");
	writeRealToBinaryFile(G, &ChiX[2*G->N], 	fileName + "_X_Fy");
	writeRealToBinaryFile(G, &ChiX[3*G->N], 	fileName + "_X_Fxy");
	
	writeRealToBinaryFile(G,  ChiY, 			fileName + "_Y_F");
	writeRealToBinaryFile(G, &ChiY[1*G->N], 	fileName + "_Y_Fx");
	writeRealToBinaryFile(G, &ChiY[2*G->N], 	fileName + "_Y_Fy");
	writeRealToBinaryFile(G, &ChiY[3*G->N], 	fileName + "_Y_Fxy");
}

/*
void writeRealToImage(TCudaGrid2D *G, ptype *var, string fileName, ptype min, ptype max, color_map_choice map, bool INVERTED)
{
	const unsigned int dimX = G->NX;
	const unsigned int dimY = G->NY;
	bitmap_image image(dimX,dimY);

	ptype var_range = max - min;

	const rgb_store *color_map;
	switch(map)
	{
		case GRAY:{
				color_map = gray_colormap;
				break;
				}
		case HOT:{
				color_map = hot_colormap;
				break;
				}
		case JET:{
				color_map = jet_colormap;
				break;
				}
	}

	int n=0;
	int var2;
	for (unsigned int y = 0; y < dimY; ++y)
	{
		for (unsigned int x = 0; x < dimX; ++x)
		{
		 
			if(INVERTED)
				var2 = (int)((max - var[n])*990.0/var_range) + 5;
			else
				var2 = (int)((var[n] - min)*990.0/var_range) + 5;
		 
		 rgb_store col = color_map[(int)(var2)];
		 image.set_pixel(x,dimY - 1 - y,col.red,col.green,col.blue);
		 n++;
		}
	}
	
	
	fileName = "data/" + fileName + ".bmp";
	image.save_image(fileName.c_str());
}
*/


/*******************************************************************
*					    Writting in binary						   *
*******************************************************************/


void writeAllRealToBinaryFile(int Len, ptype *var, string simulationName, string fileName)
{
	fileName = "data/" + simulationName + "/all_save_data/" + fileName + ".data";
	ofstream file(fileName.c_str(), ios::out | ios::binary);
	
		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
			return;
		}
	
	FOR(l, Len)
		file.write( (char*) &var[l], sizeof(ptype) );
		
	file.close();
}


void readAllRealFromBinaryFile(int Len, ptype *var, string simulationName, string fileName)
{
	fileName = "data/" + simulationName + "/all_save_data/" + fileName + ".data";
	ifstream file(fileName.c_str(), ios::in | ios::binary);
	
		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
		}
		
	FOR(l, Len)
		file.read( (char*) &var[l], sizeof(ptype) );
		
	file.close();
}


void writeAllData(TCudaGrid2D *Gc, TCudaGrid2D *Gsf, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *wsf, ptype *wc, ptype *lsf, ptype *Phi, int stack_map_passed, string t_nb, string simulationName)
{
	
	string fileName = "data/" + simulationName + "/all_save_data/stack_map_passed_" + t_nb + ".data";
	ofstream file(fileName.c_str(), ios::out | ios::binary);
	file.write( (char*) &stack_map_passed, sizeof(int) );
	file.close();
	
	
	writeAllRealToBinaryFile(Gsf->N, wsf, simulationName, "w_" + t_nb);
	writeAllRealToBinaryFile(Gc->N, wc, simulationName, "wc_" + t_nb);
	//writeAllRealToBinaryFile(Gsf->N, lsf, simulationName, "l_" + t_nb);
	writeAllRealToBinaryFile(4*Gc->N, Phi, simulationName, "Phi_" + t_nb);
	//writeAllRealToBinaryFile(4*Gc->N, ChiX, simulationName, "ChiX_" + t_nb);
	//writeAllRealToBinaryFile(4*Gc->N, ChiY, simulationName, "ChiY_" + t_nb);
	//writeAllRealToBinaryFile(4*Gc->N, ChiDualX, simulationName, "ChiDualX_" + t_nb);
	//writeAllRealToBinaryFile(4*Gc->N, ChiDualY, simulationName, "ChiDualY_" + t_nb);
	writeAllRealToBinaryFile(stack_map_passed*4*Gc->N, ChiX_stack, simulationName, "ChiX_stack_" + t_nb);
	writeAllRealToBinaryFile(stack_map_passed*4*Gc->N, ChiY_stack, simulationName, "ChiY_stack_" + t_nb);
	
}


void readAllData(TCudaGrid2D *Gc, TCudaGrid2D *Gsf, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *wsf, ptype *lsf, ptype *Phi, int stack_map_passed, string t_nb, string simulationName)
{
	
	string fileName = "data/" + simulationName + "/all_save_data/stack_map_passed_" + t_nb + ".data";
	ifstream file(fileName.c_str(), ios::in | ios::binary);
	file.read( (char*) &stack_map_passed, sizeof(int) );
	file.close();
	
	
	readAllRealFromBinaryFile(Gsf->N, wsf, simulationName, "w_" + t_nb);
	readAllRealFromBinaryFile(Gsf->N, lsf, simulationName, "l_" + t_nb);
	readAllRealFromBinaryFile(4*Gc->N, Phi, simulationName, "Phi_" + t_nb);
	readAllRealFromBinaryFile(4*Gc->N, ChiX, simulationName, "ChiX_" + t_nb);
	readAllRealFromBinaryFile(4*Gc->N, ChiY, simulationName, "ChiY_" + t_nb);
	readAllRealFromBinaryFile(4*Gc->N, ChiDualX, simulationName, "ChiDualX_" + t_nb);
	readAllRealFromBinaryFile(4*Gc->N, ChiDualY, simulationName, "ChiDualY_" + t_nb);
	readAllRealFromBinaryFile(stack_map_passed*4*Gc->N, ChiX_stack, simulationName, "ChiX_stack_" + t_nb);
	readAllRealFromBinaryFile(stack_map_passed*4*Gc->N, ChiY_stack, simulationName, "ChiY_stack_" + t_nb);
	
}


void writeRealToBinaryAnyFile(int Len, ptype *var, string fileAdress)
{
	string fileName = fileAdress;
	ofstream file(fileName.c_str(), ios::out | ios::binary);
	
		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
			return;
		}
	
	FOR(l, Len)
		file.write( (char*) &var[l], sizeof(ptype) );
		
	file.close();
}


void readRealToBinaryAnyFile(int Len, ptype *var, string fileAdress)
{
	string fileName = fileAdress;
	ifstream file(fileName.c_str(), ios::in | ios::binary);
	
		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
		}
		
	FOR(l, Len)
		file.read( (char*) &var[l], sizeof(ptype) );
		
	file.close();
}








/******************************************************************/
/*******************************************************************
*							   Old								   *
*******************************************************************/
/******************************************************************/





void writeComplexToFile(TCudaGrid2D *G, cuPtype *var, string fileName)
{
}

void writeHalfComplexToFile(TCudaGrid2D *G, cuPtype *var, string fileName)
{
}

void writeDiffeoToFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr)
{
}

void writeDiffeoStackToFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr)
{
}

int readDiffeoFromBinaryFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr)
{
return 0;
}

void writeVorticityToFile(TCudaGrid2D *G, cuPtype *w, string simulationName, string fileName, int ctr)
{
}

void writeVorticityToImage(TCudaGrid2D *G, cuPtype *w, ptype min, ptype max, string simulationName, string fileName, int ctr)
{
}

void writeHalfComplexToImage(TCudaGrid2D *G, cuPtype *var, string fileName, ptype min, ptype max, color_map_choice map, bool INVERTED)
{
}

