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

#include <algorithm>  // all_of

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
	std::string message="\n\n\n";
	message += "╔════════════════════════════════════════════╗\n";
	message += "║          Starting Vlasov 1D Simulation     ║\n" ;
    message += "╚════════════════════════════════════════════╝\n" ;
	// start clock as first thing to measure initializations too
	auto begin = std::chrono::high_resolution_clock::now();
	// clock_t begin = clock();

	/*******************************************************************
	*						 	 Constants							   *
	*******************************************************************/

	// boundary information for translating, 6 coords for 3D - (x0, x1, y0, y1, z0, z1)
	double bounds[6] = {0, 4.0*PI, -2.5*PI, 2.5*PI, 0, 0};
	SettingsMain.getDomainBounds(bounds);

	if (SettingsMain.getInitialConditionNum() == 0) {
		message += "\nInitial condition: Landau Damping\n\n";
		for (int i = 0; i < 4; ++i) bounds[i] = bounds[i]*PI; // multiply by PI
	}
	else if (SettingsMain.getInitialConditionNum() == 1){
		// Notice that for the twostream instability we only multiply the x bounds by PI
		 for (int i = 0; i < 2; ++i)	 bounds[i] = bounds[i]*PI; // multiply by PI
		 message += "\nInitial condition: Two Stream Instability\n\n";
	}
	std::cout<<message;
	printf("Domain bounds:\nx0/PI = %f, x1/PI = %f,\ny0/PI= %f, y1/PI = %f,\nz0/Pi = %f, z1/PI = %f\n", bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);

	// the code seems to work only square domains!!! is it a bug of a feature? I think its sad
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
	logger.push(message);


	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Starting memory initialization"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		logger.push(message);
	}

    // print version details
    if (SettingsMain.getVerbose() >= 3) {

		int deviceCount; cudaGetDeviceCount(&deviceCount);
		message = "Number of CUDA devices = " + to_str(deviceCount); logger.push(message);
		int activeDevice; cudaGetDevice(&activeDevice);
		message = "Active CUDA device = " + to_str(activeDevice); logger.push(message);
		int version; cudaRuntimeGetVersion(&version);
    	message = "Cuda runtime version = " + to_str(version); logger.push(message);
		cudaDriverGetVersion(&version);
    	message = "Cuda driver version = " + to_str(version); logger.push(message);
		cufftGetVersion(&version);
		message = "CuFFT version = " + to_str(version); logger.push(message);
    }

    // print simulation details
    if (SettingsMain.getVerbose() >= 1) {
		message = "Solving = " + SettingsMain.getSimulationType(); logger.push(message);
		message = "Initial condition = " + SettingsMain.getInitialCondition(); logger.push(message);
		message = "Iter max = " + to_str(iterMax); logger.push(message);
		message = "Name of simulation = " + SettingsMain.getFileName(); logger.push(message);
    }

    /*
	 *
	 * Variable creation
	 *
	 */
	std::map<std::string, CmmVar2D*> cmmVarMap;  // map containing all variables
    std::map<std::string, CmmPart*> cmmPartMap;  // map containing all particle variables

	// backwards map
	CmmVar2D ChiX(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse(), 1, bounds, -1);
	CmmVar2D ChiY(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse(), 1, bounds, -1);
	cmmVarMap["ChiX"] = &ChiX; cmmVarMap["ChiY"] = &ChiY; mb_used_RAM_GPU += ChiX.RAM_size + ChiY.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized backwards maps arrays"; logger.push(message); }

	// vorticity on coarse grid, needed only for saving
	CmmVar2D Vort(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse(), 1, bounds, 0);
	cmmVarMap["Vort"] = &Vort; mb_used_RAM_GPU += Vort.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized coars vort array"; logger.push(message); }

	// vorticity initial condition on fine grid for each sub-map
	CmmVar2D Vort_fine_init(SettingsMain.getGridFine(), SettingsMain.getGridFine(), 1, bounds, -2);
	cmmVarMap["Vort_fine_init"] = &Vort_fine_init; mb_used_RAM_GPU += Vort_fine_init.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized vort_fine_init array"; logger.push(message); }

	// discrete initial condition for vorticity, dynamically let size vanish if not needed
	int type_discrete = -10 + (-2 + 10) * SettingsMain.getInitialDiscrete();  // -2 or -10 dependend on if it is needed
	CmmVar2D Vort_discrete_init(SettingsMain.getInitialDiscreteGrid(), SettingsMain.getInitialDiscreteGrid(), 1, bounds, type_discrete);
	cmmVarMap["Vort_discrete_init"] = &Vort_discrete_init; mb_used_RAM_GPU += Vort_discrete_init.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized discrete initial array"; logger.push(message); }

	// psi map, depended on the lagrange order the length differs
	CmmVar2D Psi(SettingsMain.getGridPsi(), SettingsMain.getGridPsi(), 1, bounds, SettingsMain.getLagrangeOrder());
	cmmVarMap["Psi"] = &Psi; mb_used_RAM_GPU += Psi.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized stream function array"; logger.push(message); }

	// forward map, dynamically let size vanish if not needed
	int type_forward = -10 + (-1 + 10) * SettingsMain.getForwardMap();  // -1 or -10 dependend on if it is needed
	CmmVar2D ChiX_f(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse(), 1, bounds, type_forward);
	CmmVar2D ChiY_f(SettingsMain.getGridCoarse(), SettingsMain.getGridCoarse(), 1, bounds, type_forward);
	cmmVarMap["ChiX_f"] = &ChiX_f; cmmVarMap["ChiY_f"] = &ChiY_f; mb_used_RAM_GPU += ChiX_f.RAM_size + ChiY_f.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized forward maps arrays"; logger.push(message); }

	// additional grids or plans needed without variable attached to it
	// vorticity grid and plans do not have a variable
	CmmVar2D empty_vort(SettingsMain.getGridVort(), SettingsMain.getGridVort(), 1, bounds, -10);
	cmmVarMap["empty_vort"] = &empty_vort; mb_used_RAM_GPU += empty_vort.RAM_size;
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized empty_vort array"; logger.push(message); }

	CmmVar2D* Sample[SettingsMain.getSaveSampleNum()];
//	double *Dev_Sample; size_t size_max_sample = 0;
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); i_save++) {
		Sample[i_save] = new CmmVar2D(SettingsMain.getSaveSample()[i_save].grid, SettingsMain.getSaveSample()[i_save].grid, 1, SettingsMain.getDomainBoundsPointer(), -10);
		cmmVarMap["Sample_" + to_str(i_save)] = Sample[i_save];
		// directly free resources of this variable, as it is overwritten later
		cudaFree(Sample[i_save]->Dev_var);
		// the device memory is later included with the temporal variable
	}
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized Sample variables"; logger.push(message); }

	CmmVar2D* Zoom[SettingsMain.getSaveZoomNum()];
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); i_save++) {
		Zoom[i_save] = new CmmVar2D(SettingsMain.getSaveZoom()[i_save].grid, SettingsMain.getSaveZoom()[i_save].grid, 1, SettingsMain.getDomainBoundsPointer(), -10);
		cmmVarMap["Zoom_" + to_str(i_save)] = Zoom[i_save];
		// directly free resources of this variable, as it is overwritten later
		cudaFree(Zoom[i_save]->Dev_var);
		// the device memory is later included with the temporal variable
	}
	if (SettingsMain.getVerbose() >= 4) { message = "Initialized Zoom variables"; logger.push(message); }


	/*******************************************************************
	*				CuFFT plan space allocation						   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 	we never run ffts in parallel, so we can reuse the temp space  *
	* 																   *
	*******************************************************************/

	// compute maximum plan size
	size_t size_max_fft = 0;
	for (const auto& i_cmmVar : cmmVarMap) {
		// skip some variables if they are not used
		std::string var_key = i_cmmVar.first;
		if (var_key == "Vort_discrete_init" && !SettingsMain.getInitialDiscrete()) continue;
		if ((var_key == "ChiX_f" || var_key == "ChiY_f") && !SettingsMain.getForwardMap()) continue;

		CmmVar2D cuda_var_2D_now = *i_cmmVar.second;
		for (const auto& i_size : cuda_var_2D_now.plan_size) { size_max_fft = std::max(size_max_fft, i_size); }
	}

	// allocate work area
	void *fft_work_area;
	cudaMalloc(&fft_work_area, size_max_fft);
	mb_used_RAM_GPU += size_max_fft / 1e6;

	// set new workarea to plans
	for (const auto& i_cmmVar : cmmVarMap) {
		CmmVar2D cuda_var_2D_now = *i_cmmVar.second;
		cuda_var_2D_now.setWorkArea(fft_work_area);
	}

	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized cufft workplan"; logger.push(message);
	}


	/*******************************************************************
	*							Temporary variable
	*          Used for temporary copying and storing of data
	*                       in various locations
	*  casting (cufftDoubleReal*) makes them usable for double arrays
	*
	*  Different sizes are:
	*  		Nfft: (2 x (Nx/2 + 1) x Ny		- this is slightly larger than NReal
	*  		NReal: Nx x Ny
	*
	*  The different grid sizes checked are:
	*  		- 1 x Fine Nfft (Sub-map initial condition, used for incomp threshold check)
	*  		- 8 x Coarse NReal (Map X- and Y-direction in Hermite form, used for new map)
	*  		- 1 x Psi Nfft (used for ???)
	*  		- 1 x Vort Nfft (For evaluating stream function before shifting grid to Psi grid)
	*  		- 1 x Discrete init Nfft (We need to create Hermite form for which we need temporal array)
	*  		- 3 x Sample Nfft + 1 x PartF Num Sum (all memory usage is streamlined here), larger of the following:
	*  			2 x NReal for sampling map, 1 x NReal for storing variables
	*  			1 x NReal for storing variables, 1 x Nfft for computing derivatives
	*  		- 3 x Zoom Nfft + 1 x PartF Num Sum (all memory usage is streamlined here)
	*  			2 x NReal for sampling map, 1 x NReal for storing variables
	*  		- 2 x PartA Num (sampling velocity values from Psi for fluid particles)
	*
	*  	The temporary memory is assigned to the following variables. Check for possible clashs when using!
	*  		- Its own variable
	*  		- empty_vort
	*  		- All Sample_var
	*  		- All Zoom_var
	*
	*******************************************************************/

	// set after largest grid, checks are for failsave in case one chooses weird grid

	size_t size_max_c = std::max(Vort_fine_init.Grid->sizeNfft, Vort_fine_init.Grid->sizeNReal);
	size_max_c = std::max(size_max_c, 4*ChiX.Grid->sizeNReal + 4*ChiY.Grid->sizeNReal);
	size_max_c = std::max(size_max_c, 2*Vort.Grid->sizeNfft);  // redundant, for palinstrophy
	size_max_c = std::max(size_max_c, Psi.Grid->sizeNfft);  // Psi
	size_max_c = std::max(size_max_c, empty_vort.Grid->sizeNfft);
	size_max_c = std::max(size_max_c, empty_vort.Grid->sizeNReal);  // a bit redundant in comparison to above, but lets just include it
	if (SettingsMain.getInitialDiscrete()) {
		size_max_c = std::max(size_max_c, Vort_discrete_init.Grid->sizeNfft);
		size_max_c = std::max(size_max_c, Vort_discrete_init.Grid->sizeNReal);  // is basically redundant, but lets take it anyways
	}
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
		size_max_c = std::max(size_max_c, 3*Sample[i_save]->Grid->sizeNReal);
		size_max_c = std::max(size_max_c, 2*Sample[i_save]->Grid->sizeNfft + 1*Sample[i_save]->Grid->sizeNReal);
		if (SettingsMain.getForwardMap() and SettingsMain.getParticlesForwarded()) {
			// we need to save and transfer new particle positions somehow, in addition to map
			size_t stacked_size = 2*Sample[i_save]->Grid->sizeNReal;
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				stacked_size += 2*SettingsMain.getParticlesForwarded()[i_p].num*sizeof(double);
			}
			size_max_c = std::max(size_max_c, stacked_size);
		}
	}
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
		size_max_c = std::max(size_max_c, 3*Zoom[i_save]->Grid->sizeNReal);
		if (SettingsMain.getForwardMap() and SettingsMain.getParticlesForwarded()) {
			// we need to save and transfer new particle positions somehow, in addition to map
			size_t stacked_size = 2*Zoom[i_save]->Grid->sizeNReal;
			for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
				stacked_size += 2*SettingsMain.getParticlesForwarded()[i_p].num*sizeof(double);
			}
			size_max_c = std::max(size_max_c, stacked_size);
		}
	}
	// take care of fluid particles in order to save the velocity later if needed, normally shouldn't add restrictions
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		size_max_c = std::max(size_max_c, 2*SettingsMain.getParticlesAdvected()[i_p].num*sizeof(double));
	}
	// for now three thrash variables are needed for cufft with hermites
	cufftDoubleComplex *Dev_Temp_C1;
	cudaMalloc((void**)&Dev_Temp_C1, size_max_c);
	mb_used_RAM_GPU += size_max_c / 1e6;

	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized GPU temp array"; logger.push(message);
	}

	// set empty_vort variable device memory after temporary variable, so that I can use it
	empty_vort.Dev_var = (double*)Dev_Temp_C1;
	empty_vort.RAM_size = size_max_c / 1e6;
	// set sample variable device memory after temporary variable
	for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); i_save++) {
		Sample[i_save]->Dev_var = (double*)Dev_Temp_C1;
		Sample[i_save]->RAM_size = size_max_c / 1e6;
	}
	// set zoom variable device memory after temporary variable
	for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); i_save++) {
		Zoom[i_save]->Dev_var = (double*)Dev_Temp_C1;
		Zoom[i_save]->RAM_size = size_max_c / 1e6;
	}


	/*******************************************************************
	*		Load in Discrete initial condition for vorticity		   *
	*******************************************************************/
	// ToDo: Add discrete initial condition loading for scalar

	if (SettingsMain.getInitialDiscrete()) {
		// read in values and copy to device
		bool read_file = readTransferFromBinaryFile(Vort_discrete_init.Grid->N, Vort_discrete_init.Dev_var, SettingsMain.getInitialDiscreteLocation());

		// safety check in case we cannot read the file
		if (!read_file) {
			message = "Unable to read discrete initial condition .. exitting"; logger.push(message); exit(0);
		}

		// compute hermite version, first forward FFT and normalization
		cufftExecD2Z(Vort_discrete_init.plan_D2Z, Vort_discrete_init.Dev_var, Dev_Temp_C1);
		k_normalize_h<<<Vort_discrete_init.Grid->blocksPerGrid, Vort_discrete_init.Grid->threadsPerBlock>>>(Dev_Temp_C1, *Vort_discrete_init.Grid);

		// Hermite vorticity array : [vorticity, x-derivative, y-derivative, xy-derivative]
		fourier_hermite(*Vort_discrete_init.Grid, Dev_Temp_C1, Vort_discrete_init.Dev_var, Vort_discrete_init.plan_Z2D);

		if (SettingsMain.getVerbose() >= 1) {
			message = "Using discrete initial condition"; logger.push(message);
		}
	}


	// initialize constant random variables for random initial conditions, position here chosen arbitrarily
	double h_rand[1000];
	std::mt19937_64 rand_gen(0); std::uniform_real_distribution<double> rand_fun(-1.0, 1.0);
	for (int i_rand = 0; i_rand < 1000; i_rand++) {
		h_rand[i_rand] = rand_fun(rand_gen);
	}
	cudaMemcpyToSymbol(d_rand, h_rand, sizeof(double)*1000);

	// initialize init_param values
	cudaMemcpyToSymbol(d_init_params, SettingsMain.getInitialParamsPointer(), sizeof(double)*10);


	/*******************************************************************
	*					       Chi_stack							   *
	* 	We need to save the variable Chi to be able to make	the        *
	* 	remapping or the zoom for backwards map		 				   *
	* 																   *
	*******************************************************************/

	double map_size = 8*ChiX.Grid->sizeNReal / 1e6;  // size of one mapping in x- and y- direction
	int cpu_map_num = int(double(SettingsMain.getMemRamCpuRemaps())/map_size);  // define how many more remappings we can save on CPU than on GPU
	if (SettingsMain.getForwardMap()) cpu_map_num = (int)(cpu_map_num/2.0);  // divide by two in case of forward map

	// initialize backward map stack
	MapStack Map_Stack(ChiX.Grid, cpu_map_num);
	mb_used_RAM_GPU += 8*ChiX.Grid->sizeNReal / 1e6;
	mb_used_RAM_CPU += cpu_map_num * map_size;

	// print map details
	if (SettingsMain.getVerbose() >= 1) {
		message = "Map size in MB = " + to_str((int)map_size); logger.push(message);
		message = "Map stack length on CPU = " + to_str(cpu_map_num); logger.push(message);
	}
	if (SettingsMain.getVerbose() >= 4) {
		message = "Initialized CPU map stack"; logger.push(message);
	}


	/*******************************************************************
	*					Chi_Stack forward							   *
	*******************************************************************/

	// initialize foward map stack dynamically to be super small if we are not computing it
	TCudaGrid2D Grid_forward(1+(ChiX_f.Grid->NX-1)*SettingsMain.getForwardMap(), 1+(ChiX_f.Grid->NY-1)*SettingsMain.getForwardMap(), 1, bounds);
	MapStack Map_Stack_f(&Grid_forward, cpu_map_num);
	mb_used_RAM_GPU += 8*Grid_forward.sizeNReal / 1e6;
	mb_used_RAM_CPU += cpu_map_num * map_size * SettingsMain.getForwardMap();

	if (SettingsMain.getForwardMap() and SettingsMain.getVerbose() >= 4) {
		message = "Initialized CPU forward map stack"; logger.push(message);
	}

	// initialize forward particles position, will stay constant over time though so a dynamic approach could be chosen later too
	// however, the particles should not take up too much memory, so I guess it's okay
	CmmPart *Part_Pos_Forward[SettingsMain.getParticlesForwardedNum()];
	ParticlesForwarded* particles_forwarded = SettingsMain.getParticlesForwarded();
	for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
		int p_f_num = 1;  // set length to 1 if no forward map
		if (SettingsMain.getForwardMap()) p_f_num = particles_forwarded[i_p].num;

		// initialize particles
		Part_Pos_Forward[i_p] = new CmmPart(p_f_num, 0.0);
		mb_used_RAM_GPU += Part_Pos_Forward[i_p]->RAM_size;
		cmmPartMap["PartF_Pos_P" + to_str_0(i_p+1, 2)] = Part_Pos_Forward[i_p];

		// initialize particle position with 1 as forward particle type for parameters, *Psi.Grid for bounds
		init_particles(SettingsMain, *Part_Pos_Forward[i_p], *Psi.Grid, 1, i_p);

		// print some output to the Console
		if (SettingsMain.getVerbose() >= 1) {
			message = "Particles set F"+ to_str_0(i_p+1, 2) +" : Num = " + to_str(particles_forwarded[i_p].num, 8);
			logger.push(message);
		}
	}


	/*******************************************************************
	*							 Particles							   *
	*******************************************************************/
	// all variables have to be defined outside
	CmmPart *Part_Pos[SettingsMain.getParticlesAdvectedNum()];
	CmmPart *Part_Vel[SettingsMain.getParticlesAdvectedNum()];
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		int p_vel_num = 1;  // set length of velocity array to 1 for fluid particles
		if (particles_advected[i_p].tau != 0) p_vel_num = particles_advected[i_p].num;

		// initialize particles
		Part_Pos[i_p] = new CmmPart(particles_advected[i_p].num, particles_advected[i_p].tau);
		Part_Vel[i_p] = new CmmPart(p_vel_num, particles_advected[i_p].tau);
		mb_used_RAM_GPU += Part_Pos[i_p]->RAM_size;
		mb_used_RAM_GPU += Part_Vel[i_p]->RAM_size;
		cmmPartMap["PartA_Pos_P" + to_str_0(i_p+1, 2)] = Part_Pos[i_p];
		cmmPartMap["PartA_Vel_U" + to_str_0(i_p+1, 2)] = Part_Vel[i_p];

		// initialize particle position with 1 as forward particle type for parameters, *Psi.Grid for bounds
		init_particles(SettingsMain, *Part_Pos[i_p], *Psi.Grid, 0, i_p);

		// print some output to the Console
		if (SettingsMain.getVerbose() >= 1) {
			message = "Particles set P"+ to_str_0(i_p+1, 2) +" : Num = " + to_str(particles_advected[i_p].num, 8)
							+    " \t Tau = " + to_str(particles_advected[i_p].tau, 8);
			logger.push(message);
		}
	}

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


	// introduce part to console and print estimated gpu memory usage in mb before initialization
	if (SettingsMain.getVerbose() >= 1) {
		message = "Memory initialization finished"
				  " \t estimated GPU RAM = " + to_str(mb_used_RAM_GPU/1e3) + " gb"
		  	  	  " \t estimated CPU RAM = " + to_str(mb_used_RAM_CPU/1e3) + " gb"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		logger.push(message);
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
		logger.push(message);
	}

	//initialization of flow map as normal grid for forward and backward map
	k_init_diffeo<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var, *ChiX.Grid);

	if (SettingsMain.getForwardMap()) {
		k_init_diffeo<<<ChiX_f.Grid->blocksPerGrid, ChiX_f.Grid->threadsPerBlock>>>(ChiX_f.Dev_var, ChiY_f.Dev_var, *ChiX_f.Grid);
	}

	//setting initial conditions for vorticity by translating with initial grid
	translate_initial_condition_through_map_stack(Map_Stack, ChiX, ChiY, Vort_fine_init, Vort_discrete_init,
			Dev_Temp_C1, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete(), ID_DIST_FUNC);

	// compute first phi, stream hermite from distribution function

	evaluate_potential_from_density_hermite(SettingsMain, ChiX, ChiY, Vort_fine_init, Psi, empty_vort,
			Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

	// compute coarse vorticity as a simulation variable, is needed for entrophy and palinstrophy
	k_apply_map_and_sample_from_hermite<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var,
			Vort.Dev_var, Vort_fine_init.Dev_var, *ChiX.Grid, *ChiX.Grid, *Vort_fine_init.Grid, 0, false);
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
					logger.push(message);
				}

				// loop for all points needed, last time is reduced by one point (final order is just repetition)
				for (int i_step_init = 1; i_step_init <= i_order-(i_order==lagrange_order_init); i_step_init++) {
					// map advection, always with loop_ctr=1, direction = backwards
					advect_using_stream_hermite(SettingsMain, ChiX, ChiY, Psi,
							(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*ChiX.Grid->N, t_init, dt_init, 0, -1);
					cudaMemcpyAsync(ChiX.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1), 			   		4*ChiX.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
					cudaMemcpyAsync(ChiY.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1) + 4*ChiX.Grid->N, 4*ChiX.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
					cudaDeviceSynchronize();

					// test: incompressibility error
					double incomp_init = incompressibility_check(*Vort_fine_init.Grid, ChiX, ChiY, (cufftDoubleReal*)Dev_Temp_C1);

					//copy Psi to previous locations, from oldest-1 upwards
					for (int i_lagrange = 1; i_lagrange <= i_order-(i_order==lagrange_order_init); i_lagrange++) {
						cudaMemcpy(Psi.Dev_var + 4*Psi.Grid->N*(i_order-(i_order==lagrange_order_init)-i_lagrange+1),
								   Psi.Dev_var + 4*Psi.Grid->N*(i_order-(i_order==lagrange_order_init)-i_lagrange), 4*Psi.Grid->sizeNReal, cudaMemcpyDeviceToDevice);
					}
					// compute stream hermite from vorticity for computed time step
					evaluate_potential_from_density_hermite(SettingsMain, ChiX, ChiY, Vort_fine_init, Psi, empty_vort,
								Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

	 				// writeTransferToBinaryFile(Grid_psi.N, (cufftDoubleReal*)(Dev_Psi_real), SettingsMain, "/Stream_function_mollys", false);
	 				// error("evaluate_potential_from_density_hermite: not nted yet",134);

					// output step details to console
					if (SettingsMain.getVerbose() >= 2) {
						message = "   Init step = " + to_str(i_step_init) + "/" + to_str(i_order-(i_order==lagrange_order_init))
								+ " \t IncompErr = " + to_str(incomp_init)
								+ " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
						logger.push(message);
					}
				}

				// switch direction to always alternate
				for (int i_dt = 0; i_dt < 5; i_dt++) {
					dt_init[i_dt] = -1 * dt_init[i_dt];
					t_init[i_dt] = -1 * t_init[i_dt];
				}

				// to switch direction, we have to reverse psi arrangement too
				// outer elements - always have to be reversed / swapped
				k_swap_h<<<Psi.Grid->blocksPerGrid, Psi.Grid->threadsPerBlock>>>(Psi.Dev_var, Psi.Dev_var + 4*Psi.Grid->N*(i_order-(i_order==lagrange_order_init)), *Psi.Grid);
				// inner elements - swapped only for order = 4
				if (i_order-(i_order==lagrange_order_init) >= 3) {
					k_swap_h<<<Psi.Grid->blocksPerGrid, Psi.Grid->threadsPerBlock>>>(Psi.Dev_var + 4*Psi.Grid->N, Psi.Dev_var + 2*4*Psi.Grid->N, *Psi.Grid);
				}

				// reset map for next init direction
				k_init_diffeo<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var, *ChiX.Grid);
			}

			// reset everything for normal loop
			SettingsMain.setTimeIntegration(time_integration_init);
			SettingsMain.setLagrangeOrder(lagrange_order_init);
			k_init_diffeo<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var, *ChiX.Grid);
		}

		// initialize velocity for inertial particles
		for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
			if (particles_advected[i_p].tau != 0) {
				// set after current velocity
				if (particles_advected[i_p].init_vel) {
					Particle_advect_inertia_init<<<Part_Pos[i_p]->block, Part_Pos[i_p]->thread>>>(Part_Pos[i_p]->num,
							Part_Pos[i_p]->Dev_var, Part_Vel[i_p]->Dev_var, Psi.Dev_var, *Psi.Grid);
				}
				// set to zero
				else { cudaMemset(Part_Vel[i_p]->Dev_var, 0.0, Part_Vel[i_p]->sizeN); }
			}
		}
	}
	// restart from previous state
	else {

		// output init details to console
		if (SettingsMain.getVerbose() >= 2) {
			message = "Initializing by loading previous state";
			logger.push(message);
		}

		readMapStack(SettingsMain, Map_Stack, ChiX, ChiY, Psi, false, SettingsMain.getRestartLocation());

		// output how many maps were loaded
		if (SettingsMain.getVerbose() >= 2) {
			message = "Loaded " + to_str(Map_Stack.map_stack_ctr) + " maps to map stack";
			logger.push(message);
		}

		if (SettingsMain.getForwardMap()) {
			readMapStack(SettingsMain, Map_Stack_f, ChiX_f, ChiY_f, Psi, true, SettingsMain.getRestartLocation());
			// output how many maps were loaded
			if (SettingsMain.getVerbose() >= 2) {
				message = "Loaded " + to_str(Map_Stack.map_stack_ctr) + " maps to forward map stack";
				logger.push(message);
			}
		}

		//setting initial conditions for vorticity by translating with initial grid
		translate_initial_condition_through_map_stack(Map_Stack, ChiX, ChiY, Vort_fine_init, Vort_discrete_init,
					Dev_Temp_C1, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete(), ID_DIST_FUNC);

		// compute stream hermite with current maps
		evaluate_potential_from_density_hermite(SettingsMain, ChiX, ChiY, Vort_fine_init, Psi, empty_vort,
				Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

		// compute coarse vorticity as a simulation variable, is needed for entrophy and palinstrophy
		k_apply_map_and_sample_from_hermite<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var,
				Vort.Dev_var, Vort_fine_init.Dev_var, *ChiX.Grid, *ChiX.Grid, *Vort_fine_init.Grid, 0, false);

		// read in particles
		readParticlesState(SettingsMain, Part_Pos, Part_Vel, SettingsMain.getRestartLocation());
		if (SettingsMain.getParticlesAdvectedNum() > 0) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Loaded data for particle sets";
				logger.push(message);
			}
		}
	}

	// save function to save variables, combined so we always save in the same way and location
	save_functions(SettingsMain, logger, t0, dt, dt, Map_Stack, Map_Stack_f, cmmVarMap, cmmPartMap, Dev_Temp_C1);

	// displaying max and min of vorticity and velocity for plotting limits and cfl condition
	// vorticity minimum
	if (SettingsMain.getVerbose() >= 2) {
		thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(Vort_fine_init.Dev_var);
		double w_max = thrust::reduce(w_ptr, w_ptr + Vort_fine_init.Grid->N, 0.0, thrust::maximum<double>());
		double w_min = thrust::reduce(w_ptr, w_ptr + Vort_fine_init.Grid->N, 0.0, thrust::minimum<double>());

		// velocity minimum - we first have to compute the norm elements - problems on anthycithere so i disabled it
//		thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(Dev_W_H_fine_real);
//		thrust::device_ptr<double> temp_ptr = thrust::device_pointer_cast((cufftDoubleReal*)Dev_Temp_C1);
//
//		thrust::transform(psi_ptr + 1*Grid_psi.N, psi_ptr + 2*Grid_psi.N, psi_ptr + 2*Grid_psi.N, temp_ptr, norm_fun());
//		double u_max = thrust::reduce(temp_ptr, temp_ptr + Grid_psi.N, 0.0, thrust::maximum<double>());

//		message = "W min = " + to_str(w_min) + " - W max = " + to_str(w_max) + " - U max = " + to_str(u_max);
		message = "W min = " + to_str(w_min) + " - W max = " + to_str(w_max);
		logger.push(message);
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
		logger.push(message);
	}

	/*******************************************************************
	*							 Main loop							   *
	*******************************************************************/

	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Starting simulation loop"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		logger.push(message);
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
		//  double monitor = incompressibility_check(Grid_fine, Grid_coarse, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1);
		// printf("E-inc =%f ", monitor);
		advect_using_stream_hermite(SettingsMain, ChiX, ChiY, Psi,
				(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*ChiX.Grid->N, t_vec, dt_vec, loop_ctr, -1);
		cudaMemcpyAsync(ChiX.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1), 			   		4*ChiX.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
		cudaMemcpyAsync(ChiY.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1) + 4*ChiX.Grid->N, 4*ChiX.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
	    cudaDeviceSynchronize();

	    if (SettingsMain.getForwardMap()) {
	    	advect_using_stream_hermite(SettingsMain, ChiX_f, ChiY_f, Psi,
					(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*ChiX_f.Grid->N, t_vec, dt_vec, loop_ctr, 1);
			cudaMemcpyAsync(ChiX_f.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1), 			        4*ChiX_f.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
			cudaMemcpyAsync(ChiY_f.Dev_var, (cufftDoubleReal*)(Dev_Temp_C1) + 4*ChiX_f.Grid->N, 4*ChiX_f.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
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
	    monitor_map[0] = incompressibility_check(*Vort_fine_init.Grid, ChiX, ChiY, (cufftDoubleReal*)Dev_Temp_C1);
//		monitor_map[0] = incompressibility_check(*ChiX.Grid, ChiX, ChiY, (cufftDoubleReal*)Dev_Temp_C1);
	    if (monitor_map[0]<=0 || monitor_map[0]>1e15){
			error("Dear captain there is an incompressibility error in coarse grid. Aborting.",2345);
		}
//		monitor_map[0] = incompressibility_check(Grid_coarse, Grid_coarse, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1);

		if (SettingsMain.getForwardMap()) {
			// check forward settings only on forward grid (coarse)
			monitor_map[2] = invertibility_check(*ChiX.Grid, ChiX, ChiY, ChiX_f, ChiY_f, (cufftDoubleReal*)Dev_Temp_C1);
			monitor_map[1] = incompressibility_check(*ChiX_f.Grid, ChiX_f, ChiY_f, (cufftDoubleReal*)Dev_Temp_C1);

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
					message = "Stack Saturated : Exiting"; logger.push(message);
				}
				stack_saturated = true;
				continue_loop = false;
			}

			if (!stack_saturated) {
				if (SettingsMain.getVerbose() >= 2) {
					message = "Refining Map : Step = " + to_str(loop_ctr)
							+ " \t Maps = " + to_str(Map_Stack.map_stack_ctr)
							+ " \t Gap = " + to_str(loop_ctr - old_ctr);
					logger.push(message);
				}
				old_ctr = loop_ctr;

				//adjusting initial conditions, compute vorticity hermite
				translate_initial_condition_through_map_stack(Map_Stack, ChiX, ChiY, Vort_fine_init, Vort_discrete_init,
							Dev_Temp_C1, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete(), ID_DIST_FUNC); // 2 is for distribution function switch
				Map_Stack.copy_map_to_host(ChiX.Dev_var, ChiY.Dev_var);

				//resetting map
				k_init_diffeo<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var, *ChiX.Grid);

				if (SettingsMain.getForwardMap()) {
					Map_Stack_f.copy_map_to_host(ChiX_f.Dev_var, ChiY_f.Dev_var);
					k_init_diffeo<<<ChiX_f.Grid->blocksPerGrid, ChiX_f.Grid->threadsPerBlock>>>(ChiX_f.Dev_var, ChiY_f.Dev_var, *ChiX_f.Grid);
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
			cudaMemcpy(Psi.Dev_var + 4*Psi.Grid->N*(SettingsMain.getLagrangeOrder()-i_lagrange),
					   Psi.Dev_var + 4*Psi.Grid->N*(SettingsMain.getLagrangeOrder()-i_lagrange-1), 4*Psi.Grid->sizeNReal, cudaMemcpyDeviceToDevice);
		}

		// compute stream hermite from vorticity for next time step so that particles can benefit from it
		evaluate_potential_from_density_hermite(SettingsMain, ChiX, ChiY, Vort_fine_init, Psi, empty_vort,
				Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

		// compute coarse vorticity as a simulation variable, is needed for entrophy and palinstrophy
		k_apply_map_and_sample_from_hermite<<<ChiX.Grid->blocksPerGrid, ChiX.Grid->threadsPerBlock>>>(ChiX.Dev_var, ChiY.Dev_var,
				Vort.Dev_var, Vort_fine_init.Dev_var, *ChiX.Grid, *ChiX.Grid, *Vort_fine_init.Grid, 0, false);
		/*
		 * Particles advection after velocity update to profit from nice avaiable accelerated schemes
		 */
    	particles_advect(SettingsMain, Part_Pos, Part_Vel, Psi, t_vec, dt_vec, loop_ctr);

    	// check if starting position for particles was reached with this step and reinitialize inertial velocity to avoid stability issues
    	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
    	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
			if (particles_advected[i_p].tau != 0 && t_vec[loop_ctr_l] - particles_advected[i_p].init_time + dt*1e-5 < 0 && t_vec[loop_ctr_l + 1] - particles_advected[i_p].init_time + dt*1e-5 > 0) {
	    		// set after current velocity
	    		if (particles_advected[i_p].init_vel) {
					Particle_advect_inertia_init<<<Part_Pos[i_p]->block, Part_Pos[i_p]->thread>>>(Part_Pos[i_p]->num,
							Part_Pos[i_p]->Dev_var, Part_Vel[i_p]->Dev_var, Psi.Dev_var, *Psi.Grid);
	    		}
	    		// set to zero
	    		else { cudaMemset(Part_Vel[i_p]->Dev_var, 0.0, Part_Vel[i_p]->sizeN); }
    		}
    	}


			/*******************************************************************
			*							 Save snap shot						   *
			*				Normal variables in their respective grid
			*				Variables on sampled grid if wanted
			*				Particles together with fine particles
			*******************************************************************/
    	save_functions(SettingsMain, logger, t_vec[loop_ctr_l+1], dt_vec[loop_ctr_l+1], dt, Map_Stack, Map_Stack_f, cmmVarMap, cmmPartMap, Dev_Temp_C1);

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
				message = "Exited early : Last Cuda Error = " + to_str(cudaGetErrorName(error)); logger.push(message);
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
//					+  " \t Last Cuda Error = " + to_str(cudaGetErrorName(cudaGetLastError()))
			logger.push(message);
		}

		writeAppendToBinaryFile(1, &c_time, SettingsMain, "/Monitoring_data/Time_c");
	}

	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Simulation loop finished"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		logger.push(message);
	}


	// hackery: new loop for particles to test convergence without map influence
	if (SettingsMain.getParticlesSteps() != -1) {

		// make folder for particle time
		std::string folder_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/Particle_data/Time_C1";
		mkdir(folder_name.c_str(), 0777);

		// copy velocity to old values to be uniform and constant in time
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpyAsync(Psi.Dev_var + 4*Psi.Grid->N*i_lagrange, Psi.Dev_var, 4*Psi.Grid->sizeNReal, cudaMemcpyDeviceToDevice, streams[i_lagrange]);
		}

		// initialize particle position
		for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {

			init_particles(SettingsMain, *Part_Pos[i_p], *Psi.Grid, 0, i_p);

			// initialize particle velocity for inertial particles
	    	if (particles_advected[i_p].tau != 0) {
	    		Particle_advect_inertia_init<<<Part_Pos[i_p]->block, Part_Pos[i_p]->thread>>>(Part_Pos[i_p]->num,
							Part_Pos[i_p]->Dev_var, Part_Vel[i_p]->Dev_var, Psi.Dev_var, *Psi.Grid);
	    	}

			double dt_p = 1.0/(double)SettingsMain.getParticlesSteps();
			// initialize vectors for dt and t
			double t_p_vec[5] = {0, dt_p, 2*dt_p, 3*dt_p, 4*dt_p};
			double dt_p_vec[5] = {dt_p, dt_p, dt_p, dt_p, dt_p};
			// main loop until time 1
			auto begin_P = std::chrono::high_resolution_clock::now();
			if (SettingsMain.getVerbose() >= 2) {
				message = "Starting extra particle loop P" + to_str_0(i_p, 2) + ": \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(begin_P - begin).count()/1e6);
				logger.push(message);
			}
			for (int loop_ctr_p = 0; loop_ctr_p < SettingsMain.getParticlesSteps(); ++loop_ctr_p) {
				// particles advection
		    	particles_advect(SettingsMain, Part_Pos, Part_Vel, Psi, t_p_vec, dt_p_vec, 0, i_p);
			}

			// force synchronize after loop to wait until everything is finished
			cudaDeviceSynchronize();

			auto end_P = std::chrono::high_resolution_clock::now();
			if (SettingsMain.getVerbose() >= 2) {
				message = "Finished extra particle loop P" + to_str_0(i_p, 2) + ": \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin).count()/1e6);
				logger.push(message);
			}

			double time_p[1] = { std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin_P).count()/1e6 };
			writeAllRealToBinaryFile(1, time_p, SettingsMain, "/Monitoring_data/Time_P" + to_str_0(i_p, 2));

			// save final position
			// copy data to host
			writeTransferToBinaryFile(2*Part_Pos[i_p]->num, Part_Pos[i_p]->Dev_var, SettingsMain, "/Particle_data/Time_C1/Particles_pos_P" + to_str_0(i_p, 2), 0);
		}
	}



	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/
	save_functions(SettingsMain, logger, T_MAX, dt, dt, Map_Stack, Map_Stack_f, cmmVarMap, cmmPartMap, Dev_Temp_C1);

	// save map stack if wanted
	if (SettingsMain.getSaveMapStack()) {
		if (SettingsMain.getVerbose() >= 2) {
			message = "Saving MapStack : Maps = " + to_str(Map_Stack.map_stack_ctr)
				+ " \t Total size = " + to_str((Map_Stack.map_stack_ctr+1)*map_size)
				+ "mb \t S-time = " + to_str(t_vec[loop_ctr + SettingsMain.getLagrangeOrder() - 1], 16); logger.push(message);
		}
		writeMapStack(SettingsMain, Map_Stack, ChiX, ChiY, Psi, false);

		// save forward map stack
		if (SettingsMain.getForwardMap()) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Saving forward MapStack : Maps = " + to_str(Map_Stack.map_stack_ctr) + " \t Total size = " + to_str((Map_Stack.map_stack_ctr+1)*map_size) + "mb"; logger.push(message);
			}
			writeMapStack(SettingsMain, Map_Stack_f, ChiX_f, ChiY_f, Psi, true);
		}

		// write particle state, in case no particles, then nothing is written
		writeParticlesState(SettingsMain, Part_Pos, Part_Vel);
	}


	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/

	ChiX.free_res(); ChiY.free_res();
	Vort.free_res();
	Vort_fine_init.free_res();
	Vort_discrete_init.free_res();
	Psi.free_res();
	ChiX_f.free_res(); ChiY_f.free_res();
	empty_vort.free_res();
	cudaFree(fft_work_area);
//	cudaFree(Dev_Temp_C1);  // allready freed from empty_vort
	Map_Stack.free_res();
	Map_Stack_f.free_res();

	// I am missing Sample and Zoom plans to destroy, but maybe its not too important as their work area is already freed

	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		Part_Pos[i_p]->free_res();
		Part_Vel[i_p]->free_res();
	}
	if (SettingsMain.getForwardMap()) {
		for (int i_p = 0; i_p < SettingsMain.getParticlesForwardedNum(); ++i_p) {
			Part_Pos_Forward[i_p]->free_res();
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
		logger.push(message);
	}
}



