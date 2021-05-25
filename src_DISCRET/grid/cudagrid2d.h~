#ifndef __CUDA_GRID_2D_H__
#define __CUDA_GRID_2D_H__

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "fstream"
#include "iomanip"
#include "string.h"
#include "sstream"
#include "cstdlib"
//#include <omp.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
#include <curand.h>
//#include <curand_kernel.h>
//#include "../bitmap/bitmap_image.hpp"

using namespace std;

#ifdef __CUDACC__

	typedef double ptype; 						//float or double
	typedef cufftDoubleComplex cuPtype;			//depending on ptype

	#define BLOCK_SIZE 16	
	#define PI 		3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
	#define twoPI 	6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136
	#define FOR(i, N)	for(int i=0;i<N;i++)

	// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"
	#define PROBLEM_CODE "single_shear_layer"
	//#define TIME_TESTING																						//active le temps

	extern ptype ep;
	extern ptype epdx[4], epdy[4];

	enum color_map_choice
	{
		GRAY, 
		HOT, 
		JET
	};

	class Logger
	{
	public:
		Logger(string simulationName);
		void push(string message);
		void push();
		char buffer[1024];
		
	private:
		string fileName;
		ofstream file;
	};

	//the grid is assumed to span the space [0, NX*h] X [0, NY*h]
	class TCudaGrid2D
	{
		public:
			int NX, NY;
			long int N;
			ptype h;

			dim3 threadsPerBlock, blocksPerGrid;
			
			long int sizeNReal, sizeNComplex;
				
			TCudaGrid2D(int NX, int NY, ptype xRange);
	};

	void get_max_min(TCudaGrid2D *G, ptype *var, ptype *min, ptype *max);
	void Host_get_max_min(int len, ptype *Var_min, ptype *Var_max, ptype min, ptype max);
	__global__ void Dev_get_max_min(int len, ptype *var, ptype *min, ptype *max);

	//reading & writting functions
	void writeRealToFile(TCudaGrid2D *G, ptype *var, string fileName);
	void writeComplexToFile(TCudaGrid2D *G, cuPtype *var, string fileName);
	void writeHalfComplexToFile(TCudaGrid2D *G, cuPtype *var, string fileName);

	void writeRealToBinaryFile(TCudaGrid2D *G, ptype *var, string fileName);
	int readRealFromBinaryFile(TCudaGrid2D *G, ptype *var, string fileName);

	void writeRealToImage(TCudaGrid2D *G, ptype *var, string fileName, ptype min = 0, ptype max = 1, color_map_choice map = JET, bool INVERTED = false);
	void writeHalfComplexToImage(TCudaGrid2D *G, cuPtype *var, string fileName, ptype min = 0, ptype max = 1, color_map_choice map = JET, bool INVERTED = false);

	void writeDiffeoToFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr=0);
	void writeDiffeoStackToFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr=0);

	void writeDiffeoToBinaryFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr = 0); 
	int readDiffeoFromBinaryFile(TCudaGrid2D *G, ptype *ChiX, ptype *ChiY, string simulationName, string fileName, int ctr = 0); 

	void writeVorticityToFile(TCudaGrid2D *G, cuPtype *w, string simulationName, string fileName, int ctr = 0);
	void writeVorticityToImage(TCudaGrid2D *G, cuPtype *w, ptype min, ptype max, string simulationName, string fileName, int ctr = 0);

	const string currentDateTime();

	__global__ void kernel_fft_lap(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h);
	__global__ void kernel_fft_iLap(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h);
	__global__ void kernel_fft_dx(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h);
	__global__ void kernel_fft_dy(cuPtype *AOut, cuPtype *BOut, int NX, int NY, ptype h);








	void writeAllRealToBinaryFile(int Len, ptype *var, string simulationName, string fileName);
	void readAllRealFromBinaryFile(int Len, ptype *var, string simulationName, string fileName);
	void writeAllData(TCudaGrid2D *Gc, TCudaGrid2D *Gsf, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *wsf, ptype *wc, ptype *lsf, ptype *Phi, int stack_map_passed, string t_nb, string simulationName);
	void readAllData(TCudaGrid2D *Gc, TCudaGrid2D *Gsf, ptype *ChiX_stack, ptype *ChiY_stack, ptype *ChiX, ptype *ChiY, ptype *ChiDualX, ptype *ChiDualY, ptype *wsf, ptype *lsf, ptype *Phi, int stack_map_passed, string t_nb, string simulationName);









#endif

#endif


