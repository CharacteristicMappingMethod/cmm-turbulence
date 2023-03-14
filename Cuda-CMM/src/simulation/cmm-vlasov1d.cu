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

#include "cmm-vlasov1d.h"

#include "../numerical/cmm-particles.h"

#include "../simulation/cmm-simulation-host.h"
#include "../simulation/cmm-simulation-kernel.h"
#include "../simulation/cmm-init.h"

#include "../ui/cmm-io.h"
#include "../ui/cmm-param.h"

#include <unistd.h>
#include <chrono>
#include <math.h>
#include <random>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>  // fill elements into device vector
#include <thrust/execution_policy.h>  // for fill
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "../numerical/cmm-mesure.h"

// define constants:
#define ID_DIST_FUNC 2


extern __constant__ double d_rand[1000], d_init_params[10];  // array for random numbers being used for random initial conditions


void cuda_vlasov_1d(SettingsCMM& SettingsMain)
{
	
	// start clock as first thing to measure initializations too
	auto begin = std::chrono::high_resolution_clock::now();
	// clock_t begin = clock();

	/*******************************************************************
	*						 	 Constants							   *
	*******************************************************************/
	
	double bounds[6] = {0, 20*twoPI, -6, 6, 0, 0};						// boundary information for translating
	double t0 = SettingsMain.getRestartTime();							// time - initial
	double dt;															// time - step final
	int iterMax;														// time - maximum iteration count for safety

	double mb_used_RAM_GPU = 0, mb_used_RAM_CPU = 0;  // count memory usage by variables in mbyte

	// compute dt, for grid settings we have to use max in case we want to differ NX and NY
	if (SettingsMain.getSetDtBySteps()) dt = 1.0 / SettingsMain.getStepsPerSec();  // direct setting of timestep
	else dt = 1.0 / std::max(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse()) / SettingsMain.getFactorDtByGrid();  // setting of timestep by grid and factor
	
	// reset lagrange order, check first for particles
	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		switch (particles_advected[i_p].time_integration_num) {
			case 10: { break; }  // l_order >= 1 !
			case 20: { if (SettingsMain.getLagrangeOrder() < 2) SettingsMain.setLagrangeOrder(2); break; }
			case 30: case 31: { if (SettingsMain.getLagrangeOrder() < 3) SettingsMain.setLagrangeOrder(3); break; }
			case 40: case 41: { if (SettingsMain.getLagrangeOrder() < 4) SettingsMain.setLagrangeOrder(4); break; }
		}
	}
	if (SettingsMain.getLagrangeOverride() != -1) SettingsMain.setLagrangeOrder(SettingsMain.getLagrangeOverride());
	
	// shared parameters
	iterMax = (int)(1.05*ceil((SettingsMain.getFinalTime() - SettingsMain.getRestartTime()) / dt));  // maximum amount of steps, important for more dynamic steps, however, reduced for now
	
	// build file name together
	std::string file_name = SettingsMain.getSimName() + "_" + SettingsMain.getInitialCondition() + "_C" + to_str(SettingsMain.getGridCoarse()) + "_F" + to_str(SettingsMain.getGridFine())
			+ "_t" + to_str((int)(1.0/dt)) + "_T" + to_str(SettingsMain.getFinalTime());
	SettingsMain.setFileName(file_name);

	create_directory_structure(SettingsMain, dt, iterMax);
    Logger logger(SettingsMain);

    std::string message;  // string to be used for console output

	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Starting memory initialization"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}

    // print version details
    if (SettingsMain.getVerbose() >= 3) {

		int deviceCount; cudaGetDeviceCount(&deviceCount);
		message = "Number of CUDA devices = " + to_str(deviceCount); std::cout<<message+"\n"; logger.push(message);
		int activeDevice; cudaGetDevice(&activeDevice);
		message = "Active CUDA device = " + to_str(activeDevice); std::cout<<message+"\n"; logger.push(message);
		int version; cudaRuntimeGetVersion(&version);
    	message = "Cuda runtime version = " + to_str(version); std::cout<<message+"\n"; logger.push(message);
		cudaDriverGetVersion(&version);
    	message = "Cuda driver version = " + to_str(version); std::cout<<message+"\n"; logger.push(message);
		cufftGetVersion(&version);
		message = "CuFFT version = " + to_str(version); std::cout<<message+"\n\n"; logger.push(message);
    }

    // print simulation details
    if (SettingsMain.getVerbose() >= 1) {
		message = "Solving = " + SettingsMain.getSimulationType(); std::cout<<message+"\n"; logger.push(message);
		message = "Initial condition = " + SettingsMain.getInitialCondition(); std::cout<<message+"\n"; logger.push(message);
		message = "Iter max = " + to_str(iterMax); std::cout<<message+"\n"; logger.push(message);
		message = "Name of simulation = " + SettingsMain.getFileName(); std::cout<<message+"\n"; logger.push(message);
    }
	
	
	/*******************************************************************
	*							Grids								   *
	* 	Coarse grid where we compute derivatives				       *
	* 	Fine grid where we interpolate using Hermite Basis functions   *
	* 	Psi grid on which we compute the stream function and velocity  *
	* 	Sample grid on which we want to save and sample results        *
	*																   *
	*******************************************************************/
	
	TCudaGrid2D Grid_coarse(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse(), 1, bounds);
	TCudaGrid2D Grid_fine(SettingsMain.getGridFine(), SettingsMain.getGridFine(), 1, bounds);
	TCudaGrid2D Grid_psi(SettingsMain.getGridPsi(), SettingsMain.getGridPsi(), 1, bounds);
	TCudaGrid2D Grid_vort(SettingsMain.getGridVort(), SettingsMain.getGridVort(), 1, bounds);

	TCudaGrid2D *Grid_sample = (TCudaGrid2D *)malloc(sizeof(TCudaGrid2D) * SettingsMain.getSaveSampleNum());
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		fill_grid(Grid_sample+i_save, SettingsMain.getSaveSample()[i_save].grid, SettingsMain.getSaveSample()[i_save].grid, 1, bounds);
	}

//	TCudaGrid2D Grid_sample[SettingsMain.getSampleNum()];
//							(SettingsMain.getGridSample(), SettingsMain.getGridSample(), bounds);

	TCudaGrid2D *Grid_zoom = (TCudaGrid2D *)malloc(sizeof(TCudaGrid2D) * SettingsMain.getSaveZoomNum());
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
		fill_grid(Grid_zoom+i_save, SettingsMain.getSaveZoom()[i_save].grid, SettingsMain.getSaveZoom()[i_save].grid, 1, bounds);
	}

//	TCudaGrid2D Grid_zoom(SettingsMain.getGridZoom(), SettingsMain.getGridZoom(), bounds);
	
	TCudaGrid2D Grid_discrete(SettingsMain.getInitialDiscreteGrid(), SettingsMain.getInitialDiscreteGrid(), 1, bounds);

	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 	we never run ffts in parallel, so we can reuse the temp space  *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D;
	cufftHandle cufft_plan_fine_D2Z, cufft_plan_fine_Z2D;
	cufftHandle cufft_plan_vort_D2Z, cufft_plan_psi_Z2D;  // used for psi, but work on different grid for forward and backward
	cufftHandle* cufft_plan_sample_D2Z = (cufftHandle*)malloc(sizeof(cufftHandle) * SettingsMain.getSaveSampleNum());
	cufftHandle* cufft_plan_sample_Z2D = (cufftHandle*)malloc(sizeof(cufftHandle) * SettingsMain.getSaveSampleNum());
	cufftHandle cufft_plan_discrete_D2Z, cufft_plan_discrete_Z2D;

	// preinitialize handles
	cufftCreate(&cufft_plan_coarse_D2Z); cufftCreate(&cufft_plan_coarse_Z2D);
	cufftCreate(&cufft_plan_fine_D2Z);   cufftCreate(&cufft_plan_fine_Z2D);
	cufftCreate(&cufft_plan_vort_D2Z);   cufftCreate(&cufft_plan_psi_Z2D);
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		cufftCreate(&cufft_plan_sample_D2Z[i_save]); cufftCreate(&cufft_plan_sample_Z2D[i_save]);
	}
	cufftCreate(&cufft_plan_discrete_D2Z); cufftCreate(&cufft_plan_discrete_Z2D);

	// disable auto workspace creation for fft plans
	cufftSetAutoAllocation(cufft_plan_coarse_D2Z, 0); cufftSetAutoAllocation(cufft_plan_coarse_Z2D, 0);
	cufftSetAutoAllocation(cufft_plan_fine_D2Z, 0);   cufftSetAutoAllocation(cufft_plan_fine_Z2D, 0);
	cufftSetAutoAllocation(cufft_plan_vort_D2Z, 0);   cufftSetAutoAllocation(cufft_plan_psi_Z2D, 0);
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		cufftSetAutoAllocation(cufft_plan_sample_D2Z[i_save], 0); cufftSetAutoAllocation(cufft_plan_sample_Z2D[i_save], 0);
	}
	cufftSetAutoAllocation(cufft_plan_discrete_D2Z, 0); cufftSetAutoAllocation(cufft_plan_discrete_Z2D, 0);

	// create plans and compute needed size of each plan
	size_t workSize[8];
    cufftMakePlan2d(cufft_plan_coarse_D2Z, Grid_coarse.NX, Grid_coarse.NY, CUFFT_D2Z, &workSize[0]);
    cufftMakePlan2d(cufft_plan_coarse_Z2D, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2D, &workSize[1]);

    cufftMakePlan2d(cufft_plan_fine_D2Z,   Grid_fine.NX,   Grid_fine.NY,   CUFFT_D2Z, &workSize[2]);
    cufftMakePlan2d(cufft_plan_fine_Z2D,   Grid_fine.NX,   Grid_fine.NY,   CUFFT_Z2D, &workSize[3]);

    cufftMakePlan2d(cufft_plan_vort_D2Z,   Grid_vort.NX,   Grid_vort.NY,   CUFFT_D2Z, &workSize[4]);
    cufftMakePlan2d(cufft_plan_psi_Z2D,    Grid_psi.NX,    Grid_psi.NY,    CUFFT_Z2D, &workSize[5]);
	if (SettingsMain.getInitialDiscrete()) {
	    cufftMakePlan2d(cufft_plan_discrete_D2Z, Grid_discrete.NX,   Grid_discrete.NY,   CUFFT_D2Z, &workSize[6]);
	    cufftMakePlan2d(cufft_plan_discrete_Z2D, Grid_discrete.NX,   Grid_discrete.NY,   CUFFT_Z2D, &workSize[7]);
	}
	size_t size_max_fft = 0; size_t workSize_sample[2];
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
	    cufftMakePlan2d(cufft_plan_sample_D2Z[i_save], Grid_sample[i_save].NX,   Grid_sample[i_save].NY,   CUFFT_D2Z, &workSize_sample[0]);
	    cufftMakePlan2d(cufft_plan_sample_Z2D[i_save], Grid_sample[i_save].NX,   Grid_sample[i_save].NY,   CUFFT_Z2D, &workSize_sample[1]);
	    size_max_fft = std::max(size_max_fft, workSize_sample[0]); size_max_fft = std::max(size_max_fft, workSize_sample[1]);
	}


    // allocate memory to new workarea for cufft plans with maximum size
	for (int i_size = 0; i_size < 6; i_size++) {
		size_max_fft = std::max(size_max_fft, workSize[i_size]);
	}
	if (SettingsMain.getInitialDiscrete()) {
		size_max_fft = std::max(size_max_fft, workSize[6]); size_max_fft = std::max(size_max_fft, workSize[7]);
	}
	void *fft_work_area;
	cudaMalloc(&fft_work_area, size_max_fft);
	mb_used_RAM_GPU += size_max_fft / 1e6;

	// set new workarea to plans
	cufftSetWorkArea(cufft_plan_coarse_D2Z, fft_work_area); cufftSetWorkArea(cufft_plan_coarse_Z2D, fft_work_area);
	cufftSetWorkArea(cufft_plan_fine_D2Z, fft_work_area);   cufftSetWorkArea(cufft_plan_fine_Z2D, fft_work_area);
	cufftSetWorkArea(cufft_plan_vort_D2Z, fft_work_area);   cufftSetWorkArea(cufft_plan_psi_Z2D, fft_work_area);
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		cufftSetWorkArea(cufft_plan_sample_D2Z[i_save], fft_work_area); cufftSetWorkArea(cufft_plan_sample_Z2D[i_save], fft_work_area);
	}
	if (SettingsMain.getInitialDiscrete()) {
		cufftSetWorkArea(cufft_plan_discrete_D2Z, fft_work_area); cufftSetWorkArea(cufft_plan_discrete_Z2D, fft_work_area);
	}
	
	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized cuFFT plans"; std::cout<<message+"\n"; logger.push(message);
	}
	

	/*******************************************************************
	*							Temporary variable
	*          Used for temporary copying and storing of data
	*                       in various locations
	*  casting (cufftDoubleReal*) makes them usable for double arrays
	*******************************************************************/
	
	// set after largest grid, checks are for failsave in case one chooses weird grid
	long long int size_max_c = std::max(Grid_fine.sizeNfft, Grid_fine.sizeNReal);
	size_max_c = std::max(size_max_c, 8*Grid_coarse.sizeNReal);
	size_max_c = std::max(size_max_c, 2*Grid_coarse.sizeNfft);  // redundant, for palinstrophy
	size_max_c = std::max(size_max_c, Grid_psi.sizeNfft);
	size_max_c = std::max(size_max_c, Grid_vort.sizeNfft);
	size_max_c = std::max(size_max_c, Grid_vort.sizeNReal);  // a bit redundant in comparison to before, but lets just include it
	if (SettingsMain.getInitialDiscrete()) {
		size_max_c = std::max(size_max_c, Grid_discrete.sizeNfft);
		size_max_c = std::max(size_max_c, Grid_discrete.sizeNReal);  // is basically redundant, but lets take it anyways
	}
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		size_max_c = std::max(size_max_c, 2*Grid_sample[i_save].sizeNfft);
		size_max_c = std::max(size_max_c, 2*Grid_sample[i_save].sizeNReal);  // is basically redundant, but lets take it anyways
		if (SettingsMain.getForwardMap() and SettingsMain.getParticlesForwarded()) {
			// we need to save and transfer new particle positions somehow, in addition to map
			long long int stacked_size = 2*(Grid_sample[i_save].sizeNReal*sizeof(double));
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				stacked_size += 2*SettingsMain.getParticlesForwarded()[i_p].num*sizeof(double);
			}
			size_max_c = std::max(size_max_c, stacked_size);
		}
	}
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
		size_max_c = std::max(size_max_c, 3*Grid_zoom[i_save].sizeNReal);
//		if (SettingsMain.getZoomSavePsi()) size_max_c = std::max(size_max_c, 4*Grid_zoom.sizeNReal);
		if (SettingsMain.getForwardMap() and SettingsMain.getParticlesForwarded()) {
			// we need to save and transfer new particle positions somehow, in addition to map
			long long int stacked_size = 2*(Grid_zoom[i_save].sizeNReal*sizeof(double));
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				stacked_size += 2*SettingsMain.getParticlesForwarded()[i_p].num*sizeof(double);
			}
			size_max_c = std::max(size_max_c, stacked_size);
		}
	}
	// take care of fluid particles in order to save the velocity later if needed, normally shouldn't add restrictions
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		size_max_c = std::max(size_max_c, (long long int)(2*SettingsMain.getParticlesAdvected()[i_p].num*sizeof(double)));
	}
	// for now three thrash variables are needed for cufft with hermites
	cufftDoubleComplex *Dev_Temp_C1;
	cudaMalloc((void**)&Dev_Temp_C1, size_max_c);
	mb_used_RAM_GPU += size_max_c / 1e6;
	
	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized GPU temp array"; std::cout<<message+"\n"; logger.push(message);
	}

	
	/*******************************************************************
	*							  Chi								   *
	* 	Chi is an array that contains Chi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative   					       *
	* 																   *
	*******************************************************************/
	
	// initialize backward map
	double *Dev_ChiX, *Dev_ChiY;
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	mb_used_RAM_GPU += 8*Grid_coarse.sizeNReal / 1e6;

	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized maps"; std::cout<<message+"\n"; logger.push(message);
	}

	

	/*******************************************************************
	*					       Chi_stack							   *
	* 	We need to save the variable Chi to be able to make	the        *
	* 	remapping or the zoom for backwards map		 				   *
	* 																   *
	*******************************************************************/
	
	double map_size = 8*Grid_coarse.sizeNReal / 1e6;  // size of one mapping in x- and y- direction
	int cpu_map_num = int(double(SettingsMain.getMemRamCpuRemaps())/map_size);  // define how many more remappings we can save on CPU than on GPU
	if (SettingsMain.getForwardMap()) cpu_map_num = (int)(cpu_map_num/2.0);  // divide by two in case of forward map

	// initialize backward map stack
	MapStack Map_Stack(&Grid_coarse, cpu_map_num);
	mb_used_RAM_GPU += 8*Grid_coarse.sizeNReal / 1e6;
	mb_used_RAM_CPU += cpu_map_num * map_size;

    // print map details
    if (SettingsMain.getVerbose() >= 1) {
		message = "Map size in MB = " + to_str((int)map_size); std::cout<<message+"\n"; logger.push(message);
		message = "Map stack length on CPU = " + to_str(cpu_map_num); std::cout<<message+"\n"; logger.push(message);
    }
	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized CPU map stack"; std::cout<<message+"\n\n"; logger.push(message);
	}


	/*******************************************************************
	*					       Vorticity							   *
	* 	We need to have different variable version. coarse/fine,       *
	* 	real/complex/hat and an array that contains NE, SE, SW, NW	   *
	* 																   *
	*******************************************************************/
	
	double *Dev_W_coarse, *Dev_W_H_fine_real;
	cudaMalloc((void**)&Dev_W_coarse, Grid_coarse.sizeNReal);
	mb_used_RAM_GPU += Grid_coarse.sizeNReal / 1e6;
	//vorticity hermite
	cudaMalloc((void**)&Dev_W_H_fine_real, 3*Grid_fine.sizeNReal + Grid_fine.sizeNfft);
	Dev_W_H_fine_real += 2*Grid_fine.Nfft - Grid_fine.N; // shift to hide beginning buffer
	mb_used_RAM_GPU += (3*Grid_fine.sizeNReal + Grid_fine.sizeNfft) / 1e6;
	
	// initialize constant random variables for random initial conditions, position here chosen arbitrarily
	double h_rand[1000];
	std::mt19937_64 rand_gen(0); std::uniform_real_distribution<double> rand_fun(-1.0, 1.0);
	for (int i_rand = 0; i_rand < 1000; i_rand++) {
		h_rand[i_rand] = rand_fun(rand_gen);
	}
	cudaMemcpyToSymbol(d_rand, h_rand, sizeof(double)*1000);

	// initialize init_param values
	cudaMemcpyToSymbol(d_init_params, SettingsMain.getInitialParamsPointer(), sizeof(double)*10);

	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized coarse and fine vorticity"; std::cout<<message+"\n"; logger.push(message);
	}

	
	/*******************************************************************
	*							DISCRET								   *
	*******************************************************************/
	
	double *Dev_W_H_initial;
	cudaMalloc((void**)&Dev_W_H_initial, sizeof(double));
	if (SettingsMain.getInitialDiscrete()) {
		
		double *Host_W_initial;
		Host_W_initial = new double[Grid_discrete.N];
		cudaMalloc((void**)&Dev_W_H_initial, 3*Grid_discrete.sizeNReal + Grid_discrete.sizeNfft);
		Dev_W_H_initial += 2*Grid_discrete.Nfft - Grid_discrete.N;  // shift to hide beginning buffer
		mb_used_RAM_GPU += (3*Grid_discrete.sizeNReal + Grid_discrete.sizeNfft) / 1e6;
		
		// read in values and copy to device
		bool read_file = readAllRealFromBinaryFile(Grid_discrete.N, Host_W_initial, SettingsMain.getInitialDiscreteLocation());

		// safety check in case we cannot read the file
		if (!read_file) {
			message = "Unable to read discrete initial condition .. exitting";
			std::cout<<message+"\n"; logger.push(message);
			exit(0);
		}

		cudaMemcpy(Dev_W_H_initial, Host_W_initial, Grid_discrete.sizeNReal, cudaMemcpyHostToDevice);
		
		// compute hermite version, first forward FFT and normalization
		cufftExecD2Z(cufft_plan_discrete_D2Z, Dev_W_H_initial, Dev_Temp_C1);
		k_normalize_h<<<Grid_discrete.blocksPerGrid, Grid_discrete.threadsPerBlock>>>(Dev_Temp_C1, Grid_discrete);
		
		// Hermite vorticity array : [vorticity, x-derivative, y-derivative, xy-derivative]
		fourier_hermite(Grid_discrete, Dev_Temp_C1, Dev_W_H_initial, cufft_plan_discrete_Z2D);

		if (SettingsMain.getVerbose() >= 1) {
			message = "Using discrete initial condition";
			std::cout<<message+"\n"; logger.push(message);
		}

		if (SettingsMain.getVerbose() >= 4) {
			message = "Initialized discrete initial condition"; std::cout<<message+"\n"; logger.push(message);
		}

		delete [] Host_W_initial;
	}

	
	/*******************************************************************
	*							  Psi								   *
	* 	Psi is an array that contains Psi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative 							   *
	* 																   *
	*******************************************************************/
	
	//stream hermite on coarse computational grid, previous timesteps for lagrange interpolation included in array
	double *Dev_Psi_real;
	size_t size_psi = 3*Grid_psi.sizeNReal+Grid_psi.sizeNfft + (SettingsMain.getLagrangeOrder()-1)*4*Grid_psi.sizeNReal;
	cudaMalloc((void**) &Dev_Psi_real, size_psi);
	Dev_Psi_real += 2*Grid_psi.Nfft - Grid_psi.N;  // shift to hide beginning buffer
	mb_used_RAM_GPU += size_psi / 1e6;
	
	
	/*******************************************************************
	 * 				    	Forward map settings
	 ******************************************************************/

	// not ideal and a bit hackery, but my current initialization for the stack sucks
	TCudaGrid2D Grid_forward(1+(SettingsMain.getGridCoarse()-1)*SettingsMain.getForwardMap(),
			1+(SettingsMain.getGridCoarse()-1)*SettingsMain.getForwardMap(), 1, bounds);

	// initialize forward map
	double *Dev_ChiX_f, *Dev_ChiY_f;
	if (SettingsMain.getForwardMap()) {
		cudaMalloc((void**)&Dev_ChiX_f, 4*Grid_forward.sizeNReal);
		cudaMalloc((void**)&Dev_ChiY_f, 4*Grid_forward.sizeNReal);
		mb_used_RAM_GPU += 8*Grid_forward.sizeNReal / 1e6;

		if (SettingsMain.getVerbose() >= 4) {
			message = "Initialized forward map"; std::cout<<message+"\n"; logger.push(message);
		}
	}

	// initialize foward map stack dynamically to be super small if we are not computing it
	MapStack Map_Stack_f(&Grid_forward, cpu_map_num);
	mb_used_RAM_GPU += 8*Grid_forward.sizeNReal / 1e6;
	mb_used_RAM_CPU += cpu_map_num * map_size * SettingsMain.getForwardMap();

	if (SettingsMain.getForwardMap() and SettingsMain.getVerbose() >= 4) {
		message = "Initialized CPU forward map stack"; std::cout<<message+"\n"; logger.push(message);
	}

	// initialize forward particles position, will stay constant over time though so a dynamic approach could be chosen later too
	// however, the particles should not take up too much memory, so I guess it's okay
	int forward_particles_thread = 256;
	int *forward_particles_block = new int[SettingsMain.getParticlesForwardedNum()];  // cuda kernel settings, are constant
	double **Host_forward_particles_pos = new double*[SettingsMain.getParticlesForwardedNum()];  // position of particles
	double **Dev_forward_particles_pos = new double*[SettingsMain.getParticlesForwardedNum()];
	ParticlesForwarded* particles_forwarded = SettingsMain.getParticlesForwarded();
	if (SettingsMain.getForwardMap()) {
		for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
			forward_particles_block[i_p] = ceil(particles_forwarded[i_p].num / (double)forward_particles_thread);  // fit all particles

			// cpu memory
			Host_forward_particles_pos[i_p] = new double[2*particles_forwarded[i_p].num];
			mb_used_RAM_CPU += 2*particles_forwarded[i_p].num*sizeof(double) / 1e6;

			// gpu memory
			cudaMalloc((void**) &Dev_forward_particles_pos[i_p], 2*particles_forwarded[i_p].num*sizeof(double));
			mb_used_RAM_GPU += 2*particles_forwarded[i_p].num*sizeof(double) / 1e6;

			// initialize particle position with 1 as forward particle type for parameters
			init_particles(Dev_forward_particles_pos[i_p], SettingsMain, forward_particles_thread, forward_particles_block[i_p], bounds, 1, i_p);

			// print some output to the Console
			if (SettingsMain.getVerbose() >= 1) {
				message = "Particles set F"+ to_str_0(i_p+1, 2) +" : Num = " + to_str(particles_forwarded[i_p].num, 8);
				std::cout<<message+"\n"; logger.push(message);
			}
		}
	}


	/*******************************************************************
	*							 Particles							   *
	*******************************************************************/
	// all variables have to be defined outside
	int particle_thread = 256;
	int *particle_block = new int[SettingsMain.getParticlesAdvectedNum()];  // cuda kernel settings, are constant
	double **Host_particles = new double*[SettingsMain.getParticlesAdvectedNum()];  // position of particles
	double **Dev_particles_pos = new double*[SettingsMain.getParticlesAdvectedNum()];
	double **Dev_particles_vel = new double*[SettingsMain.getParticlesAdvectedNum()];
//	int particles_fine_old; double particles_fine_max;
//	double *Host_particles_fine_pos;

	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		particle_block[i_p] = ceil(particles_advected[i_p].num / (double)particle_thread);  // fit all particles

		// cpu memory
		Host_particles[i_p] = new double[2*particles_advected[i_p].num];
		mb_used_RAM_CPU += 2*(1+particles_advected[i_p].tau != 0)*particles_advected[i_p].num*sizeof(double) / 1e6;

		// gpu memory
		cudaMalloc((void**) &Dev_particles_pos[i_p], 2*particles_advected[i_p].num*sizeof(double));
		cudaMalloc((void**) &Dev_particles_vel[i_p], (2*particles_advected[i_p].num*(particles_advected[i_p].tau != 0) + (particles_advected[i_p].tau == 0))*sizeof(double));
		mb_used_RAM_GPU += 2*(1+particles_advected[i_p].tau != 0)*particles_advected[i_p].num*sizeof(double) / 1e6;

		// initialize particle position with 0 as advected particle type for parameters
		init_particles(Dev_particles_pos[i_p], SettingsMain, particle_thread, particle_block[i_p], bounds, 0, i_p);

		// print some output to the Console
		if (SettingsMain.getVerbose() >= 1) {
			message = "Particles set P"+ to_str_0(i_p+1, 2) +" : Num = " + to_str(particles_advected[i_p].num, 8)
							+    " \t Tau = " + to_str(particles_advected[i_p].tau, 8);
			std::cout<<message+"\n"; logger.push(message);
		}
	}


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	// all was made dynamic
	
	/*******************************************************************
	*						       Streams							   *
	*******************************************************************/
	
	const int num_streams = 5;
	cufftHandle cufftPlan_coarse_streams[num_streams], cufftPlan_fine_streams[num_streams];
	/*
	cudaStream_t streams;
	cudaStreamCreate(&streams);
	*/
	cudaStream_t streams[num_streams];
	for(int i = 0; i < num_streams; i++){
		cudaStreamCreate(&streams[i]);
		cufftSetStream(cufftPlan_coarse_streams[i], streams[i]);
		cufftSetStream(cufftPlan_fine_streams[i], streams[i]);
	}
	

	/*******************************************************************
	*	 Define variables on another, set grid for investigations 	   *
	*   We need another variable large enough to hold hermitian arrays *
	*******************************************************************/
	
	double *Dev_Temp_2;
	long int size_max_temp_2 = 0;
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		size_max_temp_2 = std::max(size_max_c, 3*Grid_sample[i_save].sizeNReal + Grid_sample[i_save].sizeNfft);
	}
	// define third temporary variable as safe buffer to be used for sample and zoom
	if (size_max_temp_2 != 0) {
		cudaMalloc((void**)&Dev_Temp_2, size_max_temp_2);
		mb_used_RAM_GPU += size_max_temp_2 / 1e6;
	}

	// introduce part to console and print estimated gpu memory usage in mb before initialization
	if (SettingsMain.getVerbose() >= 1) {
		message = "Memory initialization finished"
				  " \t estimated GPU RAM = " + to_str(mb_used_RAM_GPU/1e3) + " gb"
		  	  	  " \t estimated CPU RAM = " + to_str(mb_used_RAM_CPU/1e3) + " gb"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}

	// we are sure here, that we finished initializing, so the parameterfile can be written
	save_param_file(SettingsMain, SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/params.txt");


	/*******************************************************************
	*						   Initialization						   *
	*******************************************************************/

	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Starting simulation initialization"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}
		
	//initialization of flow map as normal grid for forward and backward map
	k_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse);

	if (SettingsMain.getForwardMap()) {
		k_init_diffeo<<<Grid_forward.blocksPerGrid, Grid_forward.threadsPerBlock>>>(Dev_ChiX_f, Dev_ChiY_f, Grid_forward);
	}

	//setting initial conditions for vorticity by translating with initial grid
	translate_initial_condition_through_map_stack(Grid_fine, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
			cufft_plan_fine_D2Z, cufft_plan_fine_Z2D, Dev_Temp_C1,
			Dev_W_H_initial, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete(),ID_DIST_FUNC); // 2 is for distribution function switch

	// compute first phi, stream hermite from distribution function
	evaluate_potential_from_density_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
			Dev_W_H_fine_real, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
			Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

	// compute coarse vorticity as a simulation variable, is needed for entrophy and palinstrophy
	k_apply_map_and_sample_from_hermite<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY,
			Dev_W_coarse, Dev_W_H_fine_real, Grid_coarse, Grid_coarse, Grid_fine, 0, false);


	/*
	 * Initialization of previous velocities for lagrange interpolation
	 * Either:  Compute backwards EulerExp for all needed velocities
	 * Or:		Increase order by back-and-forth computation until previous velocities are of desired order
	 * Or:		Load in state from previous simulation
	 */
	if (SettingsMain.getRestartTime() == 0) {
		if (SettingsMain.getLagrangeOrder() > 1) {
			// store settings for now
			std::string time_integration_init = SettingsMain.getTimeIntegration();
			int lagrange_order_init = SettingsMain.getLagrangeOrder();  // in case there was an override, we set it back to that

			// init fitting dt and t vectors, directly should work until 4th order, with right direction of first computation
			double dt_now = dt*pow(-1, lagrange_order_init);
			double t_init[5] = {0, dt_now, 2*dt_now, 3*dt_now, 4*dt_now};
			double dt_init[5] = {dt_now, dt_now, dt_now, dt_now, dt_now};

			// loop, where we increase the order each time and change the direction
			for (int i_order = 1; i_order <= lagrange_order_init; i_order++) {
				// if we only use first order, we can skip all iterations besides last one
				if (!SettingsMain.getLagrangeInitHigherOrder() and i_order != lagrange_order_init) {
					// switch direction to always alternate, done so we dont have to keep track of it
					for (int i_dt = 0; i_dt < 5; i_dt++) {
						dt_init[i_dt] = -1 * dt_init[i_dt];
						t_init[i_dt] = -1 * t_init[i_dt];
					}
					continue;
				}

				// set method, most accurate versions were chosen arbitrarily
				if (SettingsMain.getLagrangeInitHigherOrder()) {
					switch (i_order) {
						case 1: { SettingsMain.setTimeIntegration("EulerExp"); break; }
						case 2: { SettingsMain.setTimeIntegration("RK2"); break; }
						case 3: { SettingsMain.setTimeIntegration("RK3"); break; }
						case 4: { SettingsMain.setTimeIntegration("RK4"); break; }
					}
					SettingsMain.setLagrangeOrder(i_order);  // we have to set this explicitly too
				}
				else {
					SettingsMain.setTimeIntegration("EulerExp");
					SettingsMain.setLagrangeOrder(1);  // we have to set this explicitly too
				}

				// output init details to console
				if (SettingsMain.getVerbose() >= 2) {
					message = "Init order = " + to_str(i_order) + "/" + to_str(lagrange_order_init)
							+ " \t Method = " + SettingsMain.getTimeIntegration()
							+ " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
					std::cout<<message+"\n"; logger.push(message);
				}

				// loop for all points needed, last time is reduced by one point (final order is just repetition)
				for (int i_step_init = 1; i_step_init <= i_order-(i_order==lagrange_order_init); i_step_init++) {

					// map advection, always with loop_ctr=1, direction = backwards
					advect_using_stream_hermite(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
							(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N,
							Dev_Psi_real, t_init, dt_init, 0, -1);
					cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)(Dev_Temp_C1), 			   		 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
					cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
					cudaDeviceSynchronize();

					// test: incompressibility error
					double incomp_init = incompressibility_check(Grid_fine, Grid_coarse, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1);

					//copy Psi to previous locations, from oldest-1 upwards
					for (int i_lagrange = 1; i_lagrange <= i_order-(i_order==lagrange_order_init); i_lagrange++) {
						cudaMemcpy(Dev_Psi_real + 4*Grid_psi.N*(i_order-(i_order==lagrange_order_init)-i_lagrange+1),
								   Dev_Psi_real + 4*Grid_psi.N*(i_order-(i_order==lagrange_order_init)-i_lagrange), 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
					}
					// compute stream hermite from vorticity for computed time step
					evaluate_potential_from_density_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
							Dev_W_H_fine_real, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
							Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

					// output step details to console
					if (SettingsMain.getVerbose() >= 2) {
						message = "   Init step = " + to_str(i_step_init) + "/" + to_str(i_order-(i_order==lagrange_order_init))
								+ " \t IncompErr = " + to_str(incomp_init)
								+ " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
						std::cout<<message+"\n"; logger.push(message);
					}
				}

				// switch direction to always alternate
				for (int i_dt = 0; i_dt < 5; i_dt++) {
					dt_init[i_dt] = -1 * dt_init[i_dt];
					t_init[i_dt] = -1 * t_init[i_dt];
				}

				// to switch direction, we have to reverse psi arrangement too
				// outer elements - always have to be reversed / swapped
				k_swap_h<<<Grid_psi.blocksPerGrid, Grid_psi.threadsPerBlock>>>(Dev_Psi_real, Dev_Psi_real + 4*Grid_psi.N*(i_order-(i_order==lagrange_order_init)), Grid_psi);
				// inner elements - swapped only for order = 4
				if (i_order-(i_order==lagrange_order_init) >= 3) {
					k_swap_h<<<Grid_psi.blocksPerGrid, Grid_psi.threadsPerBlock>>>(Dev_Psi_real + 4*Grid_psi.N, Dev_Psi_real + 2*4*Grid_psi.N, Grid_psi);
				}

				// reset map for next init direction
				k_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse);
			}

			// reset everything for normal loop
			SettingsMain.setTimeIntegration(time_integration_init);
			SettingsMain.setLagrangeOrder(lagrange_order_init);
			k_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse);
		}

		// initialize velocity for inertial particles
		for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
			if (particles_advected[i_p].tau != 0) {
				// set after current velocity
				if (particles_advected[i_p].init_vel) {
					Particle_advect_inertia_init<<<particle_block[i_p], particle_thread>>>(particles_advected[i_p].num,
										Dev_particles_pos[i_p], Dev_particles_vel[i_p],
										Dev_Psi_real, Grid_psi);
				}
				// set to zero
				else { cudaMemset(Dev_particles_vel[i_p], 0.0, 2*particles_advected[i_p].num*sizeof(double)); }
			}
		}
	}
	// restart from previous state
	else {

		// output init details to console
		if (SettingsMain.getVerbose() >= 2) {
			message = "Initializing by loading previous state";
			std::cout<<message+"\n"; logger.push(message);
		}

		readMapStack(SettingsMain, Map_Stack, Grid_fine, Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_Psi_real, false, SettingsMain.getRestartLocation());

		// output how many maps were loaded
		if (SettingsMain.getVerbose() >= 2) {
			message = "Loaded " + to_str(Map_Stack.map_stack_ctr) + " maps to map stack";
			std::cout<<message+"\n"; logger.push(message);
		}

		if (SettingsMain.getForwardMap()) {
			readMapStack(SettingsMain, Map_Stack_f, Grid_fine, Grid_psi, Dev_ChiX_f, Dev_ChiY_f, Dev_W_H_fine_real, Dev_Psi_real, true, SettingsMain.getRestartLocation());
			// output how many maps were loaded
			if (SettingsMain.getVerbose() >= 2) {
				message = "Loaded " + to_str(Map_Stack.map_stack_ctr) + " maps to forward map stack";
				std::cout<<message+"\n"; logger.push(message);
			}
		}

		// compute stream hermite with current maps
		evaluate_potential_from_density_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
						Dev_W_H_fine_real, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
						Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

		// compute coarse vorticity as a simulation variable, is needed for entrophy and palinstrophy
		k_apply_map_and_sample_from_hermite<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY,
				Dev_W_coarse, Dev_W_H_fine_real, Grid_coarse, Grid_coarse, Grid_fine, 0, false);

		// read in particles
		readParticlesState(SettingsMain, Dev_particles_pos, Dev_particles_vel, SettingsMain.getRestartLocation());
		if (SettingsMain.getParticlesAdvectedNum() > 0) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Loaded data for particle sets";
				std::cout<<message+"\n"; logger.push(message);
			}
		}
	}


	// save function to save variables, combined so we always save in the same way and location
    // use Dev_Hat_fine for W_fine, this works because just at the end of conservation it is overwritten
	writeTimeStep(SettingsMain, t0, dt, dt, Grid_fine, Grid_coarse, Grid_psi,
			Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real,
			Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f);

	// compute conservation if wanted
	message = compute_conservation_targets(SettingsMain, t0, dt, dt, Grid_fine, Grid_coarse, Grid_psi, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1,
			cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D,
			Dev_Temp_C1);
	// output status to console
	if (SettingsMain.getVerbose() >= 3 && message != "") {
		std::cout<<message+"\n"; logger.push(message);
	}


	// sample if wanted
	message = sample_compute_and_write(SettingsMain, t0, dt, dt,
			Map_Stack, Map_Stack_f, Grid_sample, Grid_discrete, Dev_Temp_2,
			cufft_plan_sample_D2Z, cufft_plan_sample_Z2D, Dev_Temp_C1,
			Host_forward_particles_pos, Dev_forward_particles_pos, forward_particles_block, forward_particles_thread,
			Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f,
			bounds, Dev_W_H_initial);

	// output status to console
	if (SettingsMain.getVerbose() >= 3) {
		std::cout<<message+"\n"; logger.push(message);
	}

    cudaDeviceSynchronize();

    // save particle position if interested in that
    writeParticles(SettingsMain, t0, dt, dt, Dev_particles_pos, Dev_particles_vel, Grid_psi, Dev_Psi_real, (cufftDoubleReal*)Dev_Temp_C1, particle_block, particle_thread);

	// zoom if wanted, has to be done after particle initialization, maybe a bit useless at first instance
	Zoom(SettingsMain, t0, dt, dt,
			Map_Stack, Map_Stack_f, Grid_zoom, Grid_psi, Grid_discrete,
			Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f,
			(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, Dev_Psi_real,
			Host_particles, Dev_particles_pos,
			Host_forward_particles_pos, Dev_forward_particles_pos, forward_particles_block, forward_particles_thread);


	// displaying max and min of vorticity and velocity for plotting limits and cfl condition
	// vorticity minimum
	if (SettingsMain.getVerbose() >= 2) {
		thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(Dev_W_H_fine_real);
		double w_max = thrust::reduce(w_ptr, w_ptr + Grid_fine.N, 0.0, thrust::maximum<double>());
		double w_min = thrust::reduce(w_ptr, w_ptr + Grid_fine.N, 0.0, thrust::minimum<double>());

		// velocity minimum - we first have to compute the norm elements - problems on anthycithere so i disabled it
//		thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(Dev_W_H_fine_real);
//		thrust::device_ptr<double> temp_ptr = thrust::device_pointer_cast((cufftDoubleReal*)Dev_Temp_C1);
//
//		thrust::transform(psi_ptr + 1*Grid_psi.N, psi_ptr + 2*Grid_psi.N, psi_ptr + 2*Grid_psi.N, temp_ptr, norm_fun());
//		double u_max = thrust::reduce(temp_ptr, temp_ptr + Grid_psi.N, 0.0, thrust::maximum<double>());

//		message = "W min = " + to_str(w_min) + " - W max = " + to_str(w_max) + " - U max = " + to_str(u_max);
		message = "W min = " + to_str(w_min) + " - W max = " + to_str(w_max);
		std::cout<<message+"\n"; logger.push(message);
	}
	
	double* t_vec = new double[iterMax+SettingsMain.getLagrangeOrder()];
	double* dt_vec = new double[iterMax+SettingsMain.getLagrangeOrder()];

	for (int i_l = 0; i_l < SettingsMain.getLagrangeOrder(); i_l++) {
		t_vec[i_l] = t0 - (SettingsMain.getLagrangeOrder()-1 - i_l) * dt;
		dt_vec[i_l] = dt;
	}
	int loop_ctr = 0;
	int old_ctr = 0;
	bool continue_loop = true;


	// first timing save before loop - this is the initialization time
	{
		auto step = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
		writeAppendToBinaryFile(1, &diff, SettingsMain, "/Monitoring_data/Time_c");
	}

	/*******************************************************************
	*						  Last Cuda Error						   *
	*******************************************************************/
	// introduce part to console
	if (SettingsMain.getVerbose() >= 1) {
		message = "Simulation initialization finished"
				  " \t Last Cuda Error = " + to_str(cudaGetErrorName(cudaGetLastError()))
				+ " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}

	/*******************************************************************
	*							 Main loop							   *
	*******************************************************************/

	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Starting simulation loop"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}

	while(SettingsMain.getFinalTime() - t_vec[loop_ctr + SettingsMain.getLagrangeOrder()-1] > dt*1e-5 && loop_ctr < iterMax && continue_loop)
	{
		/*
		 * Timestep initialization:
		 *  - avoid overstepping final time
		 *  - avoid overstepping time targets
		 *  - save dt and t into vectors for lagrange interpolation
		 */
		int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;

		// avoid overstepping specific time targets
		double dt_now = compute_next_timestep(SettingsMain, t_vec[loop_ctr_l], dt);

		// avoid overstepping final time
		if(t_vec[loop_ctr_l] + dt_now > SettingsMain.getFinalTime()) dt_now = SettingsMain.getFinalTime() - t_vec[loop_ctr_l];
		// set new dt into time vectors for lagrange interpolation
		t_vec[loop_ctr_l + 1] = t_vec[loop_ctr_l] + dt_now;
		dt_vec[loop_ctr_l + 1] = dt_now;

		/*
		 * Map advection
		 *  - Velocity is already intialized, so we can safely do that here
		 *  - do for backward map important to flow and forward map if wanted
		 */
		advect_using_stream_hermite(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
				(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N,
				Dev_Psi_real, t_vec, dt_vec, loop_ctr, -1);
		cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)(Dev_Temp_C1), 			   		 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
		cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
	    cudaDeviceSynchronize();

	    if (SettingsMain.getForwardMap()) {
			advect_using_stream_hermite(SettingsMain, Grid_forward, Grid_psi, Dev_ChiX_f, Dev_ChiY_f,
					(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_forward.N,
					Dev_Psi_real, t_vec, dt_vec, loop_ctr, 1);
			cudaMemcpyAsync(Dev_ChiX_f, (cufftDoubleReal*)(Dev_Temp_C1), 			   		4*Grid_forward.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
			cudaMemcpyAsync(Dev_ChiY_f, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_forward.N, 4*Grid_forward.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
		    cudaDeviceSynchronize();
	    }


		/*******************************************************************
		*							 Remapping							   *
		*				Computing incompressibility
		*				Checking against threshold
		*				If exceeded: Safe map on CPU RAM
		*							 Initialize variables
		*******************************************************************/
	    double monitor_map[5]; bool forwarded_init = false;
	    monitor_map[0] = incompressibility_check(Grid_fine, Grid_coarse, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1);
//		monitor_map[0] = incompressibility_check(Grid_coarse, Grid_coarse, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1);

		if (SettingsMain.getForwardMap()) {
			monitor_map[2] = invertibility_check(Grid_coarse, Grid_coarse, Grid_forward,
					Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f, (cufftDoubleReal*)Dev_Temp_C1);

			monitor_map[1] = incompressibility_check(Grid_coarse, Grid_forward, Dev_ChiX_f, Dev_ChiY_f, (cufftDoubleReal*)Dev_Temp_C1);

			// check if we are at a starting position for forwarded particles
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				if (t_vec[loop_ctr_l+1] - particles_forwarded[i_p].init_time + dt*1e-5 < dt_vec[loop_ctr_l+1] && t_vec[loop_ctr_l+1] - particles_forwarded[i_p].init_time + dt*1e-5 >= 0) {
					forwarded_init = true;
					particles_forwarded[i_p].init_map = Map_Stack.map_stack_ctr + 1;
				}
			}
		}

		// resetting map and adding to stack if resetting condition is met
		if((monitor_map[0] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) || forwarded_init) {

			bool stack_saturated = false;
			//		if( err_incomp_b[loop_ctr] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) {
			if(Map_Stack.map_stack_ctr >= Map_Stack.cpu_map_num)
			{
				if (SettingsMain.getVerbose() >= 0) {
					message = "Stack Saturated : Exiting"; std::cout<<message+"\n"; logger.push(message);
				}
				stack_saturated = true;
				continue_loop = false;
			}

			if (!stack_saturated) {
				if (SettingsMain.getVerbose() >= 2) {
					message = "Refining Map : Step = " + to_str(loop_ctr)
							+ " \t Maps = " + to_str(Map_Stack.map_stack_ctr)
							+ " \t Gap = " + to_str(loop_ctr - old_ctr);
					std::cout<<message+"\n"; logger.push(message);
				}
				old_ctr = loop_ctr;

				//adjusting initial conditions, compute vorticity hermite
				translate_initial_condition_through_map_stack(Grid_fine, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
						cufft_plan_fine_D2Z, cufft_plan_fine_Z2D, Dev_Temp_C1,
						Dev_W_H_initial, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete(),2); // 2 is for distribution function switch

				Map_Stack.copy_map_to_host(Dev_ChiX, Dev_ChiY);

				//resetting map
				k_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse);

				if (SettingsMain.getForwardMap()) {
					Map_Stack_f.copy_map_to_host(Dev_ChiX_f, Dev_ChiY_f);
					k_init_diffeo<<<Grid_forward.blocksPerGrid, Grid_forward.threadsPerBlock>>>(Dev_ChiX_f, Dev_ChiY_f, Grid_forward);
				}
			}
		}
		// save map invetigative details
		monitor_map[3] = loop_ctr - old_ctr;
		monitor_map[4] = Map_Stack.map_stack_ctr;


	    /*
	     * Evaluation of stream hermite for the velocity at end of step to resemble velocity after timestep
	     *  - first copy the old values
	     *  - afterwards compute the new
	     *  - done before particles so that particles can benefit from nice timesteppings
	     */
        //copy Psi to previous locations, from oldest-1 upwards
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpy(Dev_Psi_real + 4*Grid_psi.N*(SettingsMain.getLagrangeOrder()-i_lagrange),
					   Dev_Psi_real + 4*Grid_psi.N*(SettingsMain.getLagrangeOrder()-i_lagrange-1), 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
		}

		// compute stream hermite from vorticity for next time step so that particles can benefit from it
		evaluate_potential_from_density_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
				Dev_W_H_fine_real, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
				Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

		// compute coarse vorticity as a simulation variable, is needed for entrophy and palinstrophy
		k_apply_map_and_sample_from_hermite<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY,
				Dev_W_coarse, Dev_W_H_fine_real, Grid_coarse, Grid_coarse, Grid_fine, 0, false);

		/*
		 * Particles advection after velocity update to profit from nice avaiable accelerated schemes
		 */
    	particles_advect(SettingsMain, Grid_psi, Dev_particles_pos, Dev_particles_vel, Dev_Psi_real,
    			t_vec, dt_vec, loop_ctr, particle_block, particle_thread);

    	// check if starting position for particles was reached with this step and reinitialize inertial velocity to avoid stability issues
    	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
    	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
			if (particles_advected[i_p].tau != 0 && t_vec[loop_ctr_l] - particles_advected[i_p].init_time + dt*1e-5 < 0 && t_vec[loop_ctr_l + 1] - particles_advected[i_p].init_time + dt*1e-5 > 0) {
	    		// set after current velocity
	    		if (particles_advected[i_p].init_vel) {
					Particle_advect_inertia_init<<<particle_block[i_p], particle_thread>>>(particles_advected[i_p].num,
										Dev_particles_pos[i_p], Dev_particles_vel[i_p],
										Dev_Psi_real, Grid_psi);
	    		}
	    		// set to zero
	    		else { cudaMemset(Dev_particles_vel[i_p], 0, 2*particles_advected[i_p].num*sizeof(double)); }
    		}
    	}


			/*******************************************************************
			*							 Save snap shot						   *
			*				Normal variables in their respective grid
			*				Variables on sampled grid if wanted
			*				Particles together with fine particles
			*******************************************************************/
		// save function to save variables, combined so we always save in the same way and location
		writeTimeStep(SettingsMain, t_vec[loop_ctr_l+1], dt_vec[loop_ctr_l+1], dt, Grid_fine, Grid_coarse, Grid_psi,
				Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real,
				Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f);

		// compute conservation if wanted
		message = compute_conservation_targets(SettingsMain, t_vec[loop_ctr_l+1], dt_vec[loop_ctr_l+1], dt, Grid_fine, Grid_coarse, Grid_psi, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1,
				cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D,
				Dev_Temp_C1);
		// output computational mesure status to console
		if (SettingsMain.getVerbose() >= 3 && message != "") {
			std::cout<<message+"\n"; logger.push(message);
		}

		// sample if wanted
		message = sample_compute_and_write(SettingsMain, t_vec[loop_ctr_l+1], dt_vec[loop_ctr_l+1], dt,
				Map_Stack, Map_Stack_f, Grid_sample, Grid_discrete, Dev_Temp_2,
				cufft_plan_sample_D2Z, cufft_plan_sample_Z2D, Dev_Temp_C1,
				Host_forward_particles_pos, Dev_forward_particles_pos, forward_particles_block, forward_particles_thread,
				Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f,
				bounds, Dev_W_H_initial);
		// output sample mesure status to console
		if (SettingsMain.getVerbose() >= 3 && message != "") {
			std::cout<<message+"\n"; logger.push(message);
		}

	    // save particle position if interested in that
	    writeParticles(SettingsMain, t_vec[loop_ctr_l+1], dt_vec[loop_ctr_l+1], dt, Dev_particles_pos, Dev_particles_vel, Grid_psi, Dev_Psi_real, (cufftDoubleReal*)Dev_Temp_C1, particle_block, particle_thread);
		
		// zoom if wanted
		Zoom(SettingsMain, t_vec[loop_ctr_l+1], dt_vec[loop_ctr_l+1], dt,
				Map_Stack, Map_Stack_f, Grid_zoom, Grid_psi, Grid_discrete,
				Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f,
				(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, Dev_Psi_real,
				Host_particles, Dev_particles_pos,
				Host_forward_particles_pos, Dev_forward_particles_pos, forward_particles_block, forward_particles_thread);

		/*
		 * Some small things at the end of the loop
		 *  - check for errors
		 *  - spit out some verbose information
		 *  - safe timing of loop
		 */

		//loop counters are increased at the end
		loop_ctr ++;

		cudaError_t error = cudaGetLastError();
		if(error != 0)
		{
			if (SettingsMain.getVerbose() >= 0) {
				message = "Exited early : Last Cuda Error = " + to_str(cudaGetErrorName(error)); std::cout<<message+"\n"; logger.push(message);
			}
			exit(0);
			break;
		}

		// append monitoring values to files
		// save timings
		writeAppendToBinaryFile(1, t_vec + loop_ctr_l + 1, SettingsMain, "/Monitoring_data/Time_s");
	    // save error for backwards incompressibility check, forwards incomp and invertibility
		writeAppendToBinaryFile(1, monitor_map, SettingsMain, "/Monitoring_data/Error_incompressibility");
		if (SettingsMain.getForwardMap()) {
			writeAppendToBinaryFile(1, monitor_map+1, SettingsMain, "/Monitoring_data/Error_incompressibility_forward");
			writeAppendToBinaryFile(1, monitor_map+2, SettingsMain, "/Monitoring_data/Error_invertibility");
		}
		// save map monitorings
		writeAppendToBinaryFile(1, monitor_map+3, SettingsMain, "/Monitoring_data/Map_gaps");
		writeAppendToBinaryFile(1, monitor_map+4, SettingsMain, "/Monitoring_data/Map_counter");

		// save timing at last part of a step to take everything into account
		double c_time;
		{
			auto step = std::chrono::high_resolution_clock::now();
			double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
			c_time = diff; // loop_ctr was already increased but first entry is init time
		}
		if (SettingsMain.getVerbose() >= 2) {
			message = "Step = " + to_str(loop_ctr)
					+ " \t S-Time = " + to_str(t_vec[loop_ctr_l+1]) + "/" + to_str(SettingsMain.getFinalTime())
					+ " \t E-inc = " + to_str(monitor_map[0]);
			if (SettingsMain.getForwardMap()) {
				message += " \t E-inc_f = " + to_str(monitor_map[1])
						+ " \t E-inv = " + to_str(monitor_map[2]);
			}
			message += " \t C-Time = " + format_duration(c_time);
			std::cout<<message+"\n"; logger.push(message);
		}

		writeAppendToBinaryFile(1, &c_time, SettingsMain, "/Monitoring_data/Time_c");
	}
	
	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Simulation loop finished"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}

	
	// hackery: new loop for particles to test convergence without map influence
	if (SettingsMain.getParticlesSteps() != -1) {

		// make folder for particle time
		string folder_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/Particle_data/Time_C1";
		mkdir(folder_name.c_str(), 0777);

		// copy velocity to old values to be uniform and constant in time
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpyAsync(Dev_Psi_real + 4*Grid_psi.N*i_lagrange, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[i_lagrange]);
		}

		// initialize particle position
		for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {

			init_particles(Dev_particles_pos[i_p], SettingsMain, particle_thread, particle_block[i_p], bounds, 0, i_p);

			// initialize particle velocity for inertial particles
	    	if (particles_advected[i_p].tau != 0) {
	    		Particle_advect_inertia_init<<<particle_block[i_p], particle_thread>>>(particles_advected[i_p].num,
	    							Dev_particles_pos[i_p], Dev_particles_vel[i_p],
	    							Dev_Psi_real, Grid_psi);
	    	}

			double dt_p = 1.0/(double)SettingsMain.getParticlesSteps();
			// initialize vectors for dt and t
			double t_p_vec[5] = {0, dt_p, 2*dt_p, 3*dt_p, 4*dt_p};
			double dt_p_vec[5] = {dt_p, dt_p, dt_p, dt_p, dt_p};
			// main loop until time 1
			auto begin_P = std::chrono::high_resolution_clock::now();
			if (SettingsMain.getVerbose() >= 2) {
				message = "Starting extra particle loop P" + to_str_0(i_p, 2) + ": \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(begin_P - begin).count()/1e6);
				std::cout<<message+"\n"; logger.push(message);
			}
			for (int loop_ctr_p = 0; loop_ctr_p < SettingsMain.getParticlesSteps(); ++loop_ctr_p) {
				// particles advection
				particles_advect(SettingsMain, Grid_psi, Dev_particles_pos, Dev_particles_vel, Dev_Psi_real,
						t_p_vec, dt_p_vec, 0, particle_block, particle_thread, i_p);
			}

			// force synchronize after loop to wait until everything is finished
			cudaDeviceSynchronize();

			auto end_P = std::chrono::high_resolution_clock::now();
			if (SettingsMain.getVerbose() >= 2) {
				message = "Finished extra particle loop P" + to_str_0(i_p, 2) + ": \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin).count()/1e6);
				std::cout<<message+"\n"; logger.push(message);
			}

			double time_p[1] = { std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin_P).count()/1e6 };
			writeAllRealToBinaryFile(1, time_p, SettingsMain, "/Monitoring_data/Time_P" + to_str_0(i_p, 2));

			// save final position
			// copy data to host
			cudaMemcpy(Host_particles[i_p], Dev_particles_pos[i_p], 2*particles_advected[i_p].num*sizeof(double), cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(2*particles_advected[i_p].num, Host_particles[i_p], SettingsMain, "/Particle_data/Time_C1/Particles_pos_P" + to_str_0(i_p, 2));
		}
	}

	
	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/

	// save function to save variables, combined so we always save in the same way and location
	writeTimeStep(SettingsMain, T_MAX, dt, dt, Grid_fine, Grid_coarse, Grid_psi,
			Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real,
			Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f);

	// compute conservation if wanted
	message = compute_conservation_targets(SettingsMain, T_MAX, dt, dt, Grid_fine, Grid_coarse, Grid_psi, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1,
			cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D,
			Dev_Temp_C1);
	// output computational mesure status to console
	if (SettingsMain.getVerbose() >= 3 && message != "") {
		std::cout<<message+"\n"; logger.push(message);
	}

	// sample if wanted
	message = sample_compute_and_write(SettingsMain, T_MAX, dt, dt,
			Map_Stack, Map_Stack_f, Grid_sample, Grid_discrete, Dev_Temp_2,
			cufft_plan_sample_D2Z, cufft_plan_sample_Z2D, Dev_Temp_C1,
			Host_forward_particles_pos, Dev_forward_particles_pos, forward_particles_block, forward_particles_thread,
			Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f,
			bounds, Dev_W_H_initial);
	// output sample mesure status to console
	if (SettingsMain.getVerbose() >= 3 && message != "") {
		std::cout<<message+"\n"; logger.push(message);
	}

    // save particle position if interested in that
    writeParticles(SettingsMain, T_MAX, dt, dt, Dev_particles_pos, Dev_particles_vel, Grid_psi, Dev_Psi_real, (cufftDoubleReal*)Dev_Temp_C1, particle_block, particle_thread);

	// zoom if wanted
	Zoom(SettingsMain, T_MAX, dt, dt,
			Map_Stack, Map_Stack_f, Grid_zoom, Grid_psi, Grid_discrete,
			Dev_ChiX, Dev_ChiY, Dev_ChiX_f, Dev_ChiY_f,
			(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, Dev_Psi_real,
			Host_particles, Dev_particles_pos,
			Host_forward_particles_pos, Dev_forward_particles_pos, forward_particles_block, forward_particles_thread);

	
	// save map stack if wanted
	if (SettingsMain.getSaveMapStack()) {
		if (SettingsMain.getVerbose() >= 2) {
			message = "Saving MapStack : Maps = " + to_str(Map_Stack.map_stack_ctr)
				+ " \t Total size = " + to_str((Map_Stack.map_stack_ctr+1)*map_size)
				+ "mb \t S-time = " + to_str(t_vec[loop_ctr + SettingsMain.getLagrangeOrder() - 1], 16); std::cout<<message+"\n"; logger.push(message);
		}
		writeMapStack(SettingsMain, Map_Stack, Grid_fine, Grid_psi,
				Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_Psi_real, false);

		// save forward map stack
		if (SettingsMain.getForwardMap()) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Saving forward MapStack : Maps = " + to_str(Map_Stack.map_stack_ctr) + " \t Total size = " + to_str((Map_Stack.map_stack_ctr+1)*map_size) + "mb"; std::cout<<message+"\n"; logger.push(message);
			}
			writeMapStack(SettingsMain, Map_Stack_f, Grid_fine, Grid_psi,
					Dev_ChiX_f, Dev_ChiY_f, Dev_W_H_fine_real, Dev_Psi_real, true);
		}

		// write particle state, in case no particles, then nothing is written
		writeParticlesState(SettingsMain, Dev_particles_pos, Dev_particles_vel);
	}

	
	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/

	cudaFree(Dev_W_H_initial);
	
	// Trash variable
	cudaFree(Dev_Temp_C1);
	if (size_max_temp_2 != 0) cudaFree(Dev_Temp_2);
	
	// Chi
	cudaFree(Dev_ChiX); cudaFree(Dev_ChiY);

	// Chistack
	Map_Stack.free_res();
	
	// Vorticity
	cudaFree(Dev_W_coarse);
	cudaFree(Dev_W_H_fine_real + Grid_fine.N - 2*Grid_fine.Nfft);  // including reshift
	
	if (SettingsMain.getInitialDiscrete()) {
		cudaFree(Dev_W_H_initial);
	}

	// Psi with reshift
	cudaFree(Dev_Psi_real + Grid_psi.N - 2*Grid_psi.Nfft);

	// CuFFT plans
	cufftDestroy(cufft_plan_coarse_D2Z); cufftDestroy(cufft_plan_coarse_Z2D);
	cufftDestroy(cufft_plan_fine_D2Z);   cufftDestroy(cufft_plan_fine_Z2D);
	cufftDestroy(cufft_plan_vort_D2Z);   cufftDestroy(cufft_plan_psi_Z2D);
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		cufftDestroy(cufft_plan_sample_D2Z[i_save]); cufftDestroy(cufft_plan_sample_Z2D[i_save]);
	}
	cudaFree(fft_work_area);

	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		delete [] Host_particles[i_p];
	    cudaFree(Dev_particles_pos[i_p]);
        cudaFree(Dev_particles_vel[i_p]);
	}

	if (SettingsMain.getForwardMap()) {
		for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
			delete [] Host_forward_particles_pos[i_p];
		    cudaFree(Dev_forward_particles_pos[i_p]);
		}
	}

	// delete monitoring values
	delete [] t_vec, dt_vec;

	// save timing at last part of a step to take everything into account
    {
		auto step = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
		writeAppendToBinaryFile(1, &diff, SettingsMain, "/Monitoring_data/Time_c");
    }

	// introduce part to console
	if (SettingsMain.getVerbose() >= 1) {
		message = "Finished simulation"
				  " \t Last Cuda Error = " + to_str(cudaGetErrorName(cudaGetLastError()))
				+ " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}
}



