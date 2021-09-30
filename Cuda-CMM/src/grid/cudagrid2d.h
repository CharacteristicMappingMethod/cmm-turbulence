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
#include <curand_kernel.h>
//#include "../bitmap/bitmap_image.hpp"

using namespace std;

#ifdef __CUDACC__

	#define BLOCK_SIZE 16	// most devices have maximum thread size of 1024, i don't know why this is 16*16=256
	#define PI 		3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067
	#define twoPI 	6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136

	#define sm_50  // for Julius, because he has a shitty graphics card
	// #define skip_remapping  // in case we want to skip the remapping for convergence purposes

	//#define DISCRET

	//active le temps
	//#define TIME_TESTING

    // #define PARTICLES
    // #define RKThree_PARTICLES

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
			double h;

			dim3 threadsPerBlock, blocksPerGrid;
			
			long int sizeNReal, sizeNComplex;
				
			TCudaGrid2D(int NX, int NY, double xRange);
	};

	void get_max_min(TCudaGrid2D *G, double *var, double *min, double *max);
	void Host_get_max_min(int len, double *Var_min, double *Var_max, double *min, double *max);
	__global__ void Dev_get_max_min(int len, double *var, double *min, double *max);

	//reading & writting functions
//	void writeRealToFile(TCudaGrid2D *G, double *var, string fileName);
//	void writeComplexToFile(TCudaGrid2D *G, cufftDoubleComplex *var, string fileName);
//	void writeHalfComplexToFile(TCudaGrid2D *G, cufftDoubleComplex *var, string fileName);
//
//	void writeRealToBinaryFile(TCudaGrid2D *G, double *var, string fileName);
//	int readRealFromBinaryFile(TCudaGrid2D *G, double *var, string fileName);
//
//	void writeRealToImage(TCudaGrid2D *G, double *var, string fileName, double min = 0, double max = 1, color_map_choice map = JET, bool INVERTED = false);
//	void writeHalfComplexToImage(TCudaGrid2D *G, cufftDoubleComplex *var, string fileName, double min = 0, double max = 1, color_map_choice map = JET, bool INVERTED = false);
//
//	void writeDiffeoToFile(TCudaGrid2D *G, double *ChiX, double *ChiY, string simulationName, string fileName, int ctr=0);
//	void writeDiffeoStackToFile(TCudaGrid2D *G, double *ChiX, double *ChiY, string simulationName, string fileName, int ctr=0);
//
//	void writeDiffeoToBinaryFile(TCudaGrid2D *G, double *ChiX, double *ChiY, string simulationName, string fileName, int ctr = 0);
//	int readDiffeoFromBinaryFile(TCudaGrid2D *G, double *ChiX, double *ChiY, string simulationName, string fileName, int ctr = 0);

//	void writeVorticityToFile(TCudaGrid2D *G, cufftDoubleComplex *w, string simulationName, string fileName, int ctr = 0);
//	void writeVorticityToImage(TCudaGrid2D *G, cufftDoubleComplex *w, double min, double max, string simulationName, string fileName, int ctr = 0);

	const string currentDateTime();

	__global__ void kernel_fft_lap(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);
	__global__ void kernel_fft_iLap(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);
	__global__ void kernel_fft_dx(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);
	__global__ void kernel_fft_dy(cufftDoubleComplex *val_in, cufftDoubleComplex *val_out, int NX, int NY, double h);








	void writeAllRealToBinaryFile(int Len, double *var, string workspace, string simulationName, string fileName);
	void readAllRealFromBinaryFile(int Len, double *var, string workspace, string simulationName, string fileName);
//	void writeAllData(TCudaGrid2D *Gc, TCudaGrid2D *Gsf, double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ChiDualX, double *ChiDualY, double *wsf, double *wc, double *lsf, double *Phi, int stack_map_passed, string t_nb, string simulationName);
//	void readAllData(TCudaGrid2D *Gc, TCudaGrid2D *Gsf, double *ChiX_stack, double *ChiY_stack, double *ChiX, double *ChiY, double *ChiDualX, double *ChiDualY, double *wsf, double *lsf, double *Phi, int stack_map_passed, string t_nb, string simulationName);
//	void writeRealToBinaryAnyFile(int Len, double *var, string fileAdress);
//	void readRealToBinaryAnyFile(int Len, double *var, string fileAdress);








#endif

#endif


