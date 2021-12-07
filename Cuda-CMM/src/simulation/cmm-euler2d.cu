#include "cmm-euler2d.h"

#include <curand.h>
#include <curand_kernel.h>

#include "../numerical/cmm-particles.h"

#include "cmm-simulation-host.h"
#include "cmm-simulation-kernel.h"

#include "../ui/cmm-io.h"
#include "../ui/cmm-param.h"

#include <unistd.h>
#include <chrono>
#include <math.h>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "../numerical/cmm-mesure.h"


void cuda_euler_2d(SettingsCMM& SettingsMain)
{
	
	// start clock as first thing to measure initializations too
	auto begin = std::chrono::high_resolution_clock::now();
	// clock_t begin = clock();

	/*******************************************************************
	*						 	 Constants							   *
	*******************************************************************/
	
	double LX = twoPI;													// domain length
	double LY = twoPI;													// domain length
	double bounds[4] = {0, LX, 0, LY};									// boundary information for translating
	int NX_coarse = SettingsMain.getGridCoarse();						// coarse grid size
	int NY_coarse = SettingsMain.getGridCoarse();						// coarse grid size
	int NX_fine = SettingsMain.getGridFine();							// fine grid size
	int NY_fine = SettingsMain.getGridFine();							// fine grid size
	
	double t0 = 0.0;													// time - initial
	double tf = SettingsMain.getFinalTime();							// time - final
	double grid_by_time = SettingsMain.getFactorDtByGrid();				// time - factor by grid
	int steps_per_sec = SettingsMain.getStepsPerSec();					// time - steps per second
	double dt;															// time - step final
	int iterMax;														// time - maximum iteration count for safety

	std::string workspace = SettingsMain.getWorkspace();						// folder where we work in
	std::string sim_name = SettingsMain.getSimName();						// name of the simulation
	std::string initial_condition = SettingsMain.getInitialCondition();		// name of the initial condition
	int snapshots_per_second = SettingsMain.getSnapshotsPerSec();		// saves per second

	double mb_used_RAM_GPU, mb_used_RAM_CPU;  // count memory usage by variables in mbyte

	// compute dt, for grid settings we have to use max in case we want to differ NX and NY
	if (SettingsMain.getSetDtBySteps()) dt = 1.0 / steps_per_sec;  // direct setting of timestep
	else dt = 1.0 / std::max(NX_coarse, NY_coarse) / grid_by_time;  // setting of timestep by grid and factor
	
	// reset lagrange order
	if (SettingsMain.getLagrangeOverride() != -1) SettingsMain.setLagrangeOrder(SettingsMain.getLagrangeOverride());
	
	// shared parameters
	iterMax = (int)(1.05*ceil(tf / dt));  // maximum amount of steps, important for more dynamic steps, however, reduced for now

	double map_size = 8*NX_coarse*NY_coarse*sizeof(double) / 1e6;  // size of one mapping
	int Nb_array_RAM = 4;  // fixed for four different stacks
	int cpu_map_num = int(double(SettingsMain.getMemRamCpuRemaps())/map_size/double(Nb_array_RAM));  // define how many more remappings we can save on CPU than on GPU
	
	// build file name together
	std::string file_name = sim_name + "_" + initial_condition + "_C" + to_str(NX_coarse) + "_F" + to_str(NX_fine) + "_t" + to_str(1.0/dt) + "_T" + to_str(tf);
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
		int version;
		cudaRuntimeGetVersion(&version);
    	message = "Cuda runtime version = " + to_str(version); std::cout<<message+"\n"; logger.push(message);
		cudaDriverGetVersion(&version);
    	message = "Cuda driver version = " + to_str(version); std::cout<<message+"\n"; logger.push(message);
		cufftGetVersion(&version);
		message = "CuFFT version = " + to_str(version); std::cout<<message+"\n\n"; logger.push(message);
    }

    // print simulation details
    if (SettingsMain.getVerbose() >= 1) {
		message = "Initial condition = " + initial_condition; std::cout<<message+"\n"; logger.push(message);
		message = "Iter max = " + to_str(iterMax); std::cout<<message+"\n"; logger.push(message);
		message = "Map stack length on CPU = " + to_str(cpu_map_num); std::cout<<message+"\n"; logger.push(message);
		message = "Map stack length total on CPU = " + to_str(cpu_map_num * Nb_array_RAM); std::cout<<message+"\n"; logger.push(message);
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
	
	TCudaGrid2D Grid_coarse(NX_coarse, NY_coarse, bounds);
	TCudaGrid2D Grid_fine(NX_fine, NY_fine, bounds);
	TCudaGrid2D Grid_psi(SettingsMain.getGridPsi(), SettingsMain.getGridPsi(), bounds);
	TCudaGrid2D Grid_vort(SettingsMain.getGridVort(), SettingsMain.getGridVort(), bounds);

	TCudaGrid2D Grid_sample(SettingsMain.getGridSample(), SettingsMain.getGridSample(), bounds);
	TCudaGrid2D Grid_zoom(SettingsMain.getGridZoom(), SettingsMain.getGridZoom(), bounds);
	
	TCudaGrid2D Grid_discrete(SettingsMain.getInitialDiscreteGrid(), SettingsMain.getInitialDiscreteGrid(), bounds);

	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 	we never run ffts in parallel, so we can reuse the temp space  *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D;
	cufftHandle cufft_plan_fine_D2Z, cufft_plan_fine_Z2D;
	cufftHandle cufft_plan_vort_D2Z, cufft_plan_psi_Z2D;  // used for psi, but work on different grid for forward and backward
	cufftHandle cufft_plan_sample_D2Z, cufft_plan_sample_Z2D;
	cufftHandle cufft_plan_discrete_D2Z, cufft_plan_discrete_Z2D;

	// preinitialize handles
	cufftCreate(&cufft_plan_coarse_D2Z); cufftCreate(&cufft_plan_coarse_Z2D);
	cufftCreate(&cufft_plan_fine_D2Z);   cufftCreate(&cufft_plan_fine_Z2D);
	cufftCreate(&cufft_plan_vort_D2Z);   cufftCreate(&cufft_plan_psi_Z2D);
	cufftCreate(&cufft_plan_sample_D2Z); cufftCreate(&cufft_plan_sample_Z2D);
	cufftCreate(&cufft_plan_discrete_D2Z); cufftCreate(&cufft_plan_discrete_Z2D);

	// disable auto workspace creation for fft plans
	cufftSetAutoAllocation(cufft_plan_coarse_D2Z, 0); cufftSetAutoAllocation(cufft_plan_coarse_Z2D, 0);
	cufftSetAutoAllocation(cufft_plan_fine_D2Z, 0);   cufftSetAutoAllocation(cufft_plan_fine_Z2D, 0);
	cufftSetAutoAllocation(cufft_plan_vort_D2Z, 0);   cufftSetAutoAllocation(cufft_plan_psi_Z2D, 0);
	cufftSetAutoAllocation(cufft_plan_sample_D2Z, 0); cufftSetAutoAllocation(cufft_plan_sample_Z2D, 0);
	cufftSetAutoAllocation(cufft_plan_discrete_D2Z, 0); cufftSetAutoAllocation(cufft_plan_discrete_Z2D, 0);

	// create plans and compute needed size of each plan
	size_t workSize[10];
    cufftMakePlan2d(cufft_plan_coarse_D2Z, Grid_coarse.NX, Grid_coarse.NY, CUFFT_D2Z, &workSize[0]);
    cufftMakePlan2d(cufft_plan_coarse_Z2D, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2D, &workSize[1]);

    cufftMakePlan2d(cufft_plan_fine_D2Z,   Grid_fine.NX,   Grid_fine.NY,   CUFFT_D2Z, &workSize[2]);
    cufftMakePlan2d(cufft_plan_fine_Z2D,   Grid_fine.NX,   Grid_fine.NY,   CUFFT_Z2D, &workSize[3]);

    cufftMakePlan2d(cufft_plan_vort_D2Z,   Grid_vort.NX,   Grid_vort.NY,   CUFFT_D2Z, &workSize[4]);
    cufftMakePlan2d(cufft_plan_psi_Z2D,    Grid_psi.NX,    Grid_psi.NY,    CUFFT_Z2D, &workSize[5]);
	if (SettingsMain.getSampleOnGrid()) {
	    cufftMakePlan2d(cufft_plan_sample_D2Z, Grid_sample.NX,   Grid_sample.NY,   CUFFT_D2Z, &workSize[6]);
	    cufftMakePlan2d(cufft_plan_sample_Z2D, Grid_sample.NX,   Grid_sample.NY,   CUFFT_Z2D, &workSize[7]);
	}
	if (SettingsMain.getInitialDiscrete()) {
	    cufftMakePlan2d(cufft_plan_discrete_D2Z, Grid_discrete.NX,   Grid_discrete.NY,   CUFFT_D2Z, &workSize[8]);
	    cufftMakePlan2d(cufft_plan_discrete_Z2D, Grid_discrete.NX,   Grid_discrete.NY,   CUFFT_Z2D, &workSize[9]);
	}

    // allocate memory to new workarea for cufft plans with maximum size
	size_t size_max_fft = 0;
	for (int i_size = 0; i_size < 6; i_size++) {
		size_max_fft = std::max(size_max_fft, workSize[i_size]);
	}
	if (SettingsMain.getSampleOnGrid()) {
		size_max_fft = std::max(size_max_fft, workSize[6]); size_max_fft = std::max(size_max_fft, workSize[7]);
	}
	if (SettingsMain.getInitialDiscrete()) {
		size_max_fft = std::max(size_max_fft, workSize[8]); size_max_fft = std::max(size_max_fft, workSize[9]);
	}
	void *fft_work_area;
	cudaMalloc(&fft_work_area, size_max_fft);
	mb_used_RAM_GPU = size_max_fft / 1e6;

	// set new workarea to plans
	cufftSetWorkArea(cufft_plan_coarse_D2Z, fft_work_area); cufftSetWorkArea(cufft_plan_coarse_Z2D, fft_work_area);
	cufftSetWorkArea(cufft_plan_fine_D2Z, fft_work_area);   cufftSetWorkArea(cufft_plan_fine_Z2D, fft_work_area);
	cufftSetWorkArea(cufft_plan_vort_D2Z, fft_work_area);   cufftSetWorkArea(cufft_plan_psi_Z2D, fft_work_area);
	if (SettingsMain.getSampleOnGrid()) {
		cufftSetWorkArea(cufft_plan_sample_D2Z, fft_work_area); cufftSetWorkArea(cufft_plan_sample_Z2D, fft_work_area);
	}
	if (SettingsMain.getInitialDiscrete()) {
		cufftSetWorkArea(cufft_plan_discrete_D2Z, fft_work_area); cufftSetWorkArea(cufft_plan_discrete_Z2D, fft_work_area);
	}
	
	
	/*******************************************************************
	*							Temporary variable
	*          Used for temporary copying and storing of data
	*                       in various locations
	*  casting (cufftDoubleReal*) makes them usable for double arrays
	*******************************************************************/
	
	// set after largest grid, checks are for failsave in case one chooses weird grid
	long int size_max_c = std::max(Grid_fine.sizeNfft, Grid_fine.sizeNReal);
	size_max_c = std::max(size_max_c, 8*Grid_coarse.sizeNReal);
	size_max_c = std::max(size_max_c, 2*Grid_coarse.sizeNfft);  // redundant, for palinstrophy
	size_max_c = std::max(size_max_c, Grid_psi.sizeNfft);
	size_max_c = std::max(size_max_c, Grid_vort.sizeNfft);
	size_max_c = std::max(size_max_c, Grid_vort.sizeNReal);  // a bit redundant in comparison to before, but lets just include it
	if (SettingsMain.getInitialDiscrete()) {
		size_max_c = std::max(size_max_c, Grid_discrete.sizeNfft);
		size_max_c = std::max(size_max_c, Grid_discrete.sizeNReal);  // is basically redundant, but lets take it anyways
	}
	if (SettingsMain.getSampleOnGrid()) {
		size_max_c = std::max(size_max_c, 2*Grid_sample.sizeNfft);
		size_max_c = std::max(size_max_c, 2*Grid_sample.sizeNReal);  // is basically redundant, but lets take it anyways
	}
	if (SettingsMain.getZoom()) {
		size_max_c = std::max(size_max_c, 3*Grid_zoom.sizeNReal);
//		if (SettingsMain.getZoomSavePsi()) size_max_c = std::max(size_max_c, 4*Grid_zoom.sizeNReal);
	}
	// for now three thrash variables are needed for cufft with hermites
	cufftDoubleComplex *Dev_Temp_C1;
	cudaMalloc((void**)&Dev_Temp_C1, size_max_c);
	mb_used_RAM_GPU += size_max_c / 1e6;
	
	// we actually only need one host array, as we always just copy from and to this and never really read files
	long int size_max_r = std::max(4*Grid_fine.N, 4*Grid_psi.N);
	size_max_r = std::max(size_max_r, 4*Grid_coarse.N);
	if (SettingsMain.getSampleOnGrid()) { size_max_r = std::max(size_max_r, 4*Grid_sample.N); }
	if (SettingsMain.getZoom()) { size_max_r = std::max(size_max_r, 4*Grid_zoom.N); }
	double *Host_save;
	Host_save = new double[size_max_r];
	mb_used_RAM_CPU = size_max_r*sizeof(double) / 1e6;

	
	/*******************************************************************
	*							  Chi								   *
	* 	Chi is an array that contains Chi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative   					       *
	* 																   *
	*******************************************************************/
	
	double *Dev_ChiX, *Dev_ChiY;
	
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	mb_used_RAM_GPU += 8*Grid_coarse.sizeNReal / 1e6;
	
	
	/*******************************************************************
	*					       Chi_stack							   *
	* 	We need to save the variable Chi to be able to make	the        *
	* 	remapping or the zoom				   					       *
	* 																   *
	*******************************************************************/
	
	MapStack Map_Stack(&Grid_coarse, cpu_map_num);
	mb_used_RAM_GPU += 8*Grid_coarse.sizeNReal / 1e6;
	mb_used_RAM_CPU += SettingsMain.getMemRamCpuRemaps();
	
	
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
	
	
	/*******************************************************************
	*							DISCRET								   *
	*******************************************************************/
	
	double *Dev_W_H_initial;
	cudaMalloc((void**)&Dev_W_H_initial, sizeof(double));
	if (SettingsMain.getInitialDiscrete()) {
		
		double *Host_W_initial;
		Host_W_initial = new double[4*Grid_discrete.N];
		cudaMalloc((void**)&Dev_W_H_initial, 4*Grid_discrete.sizeNReal);
		mb_used_RAM_GPU += (4*Grid_discrete.sizeNReal) / 1e6;
		
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
	*							 Particles							   *
	*******************************************************************/
	// all variables have to be defined outside
	int Nb_particles, particle_thread, particle_block;
	double *Host_particles_pos ,*Dev_particles_pos;  // position of particles
	double *Host_particles_vel ,*Dev_particles_vel;  // velocity of particles
	curandGenerator_t prng;
	int particles_fine_old; double particles_fine_max;
	double *Host_particles_fine_pos;
	// now set the variables if we have particles
	if (SettingsMain.getParticles()) {
		// initialize all memory
		Nb_particles = SettingsMain.getParticlesNum();
		particle_thread =  256;  // threads for particles, seems good like that
		particle_block = ceil(Nb_particles / (double)particle_thread);  // fit all particles
		Host_particles_pos = new double[2*Nb_particles*SettingsMain.getParticlesTauNum()];
		Host_particles_vel = new double[2*Nb_particles*SettingsMain.getParticlesTauNum()];
		mb_used_RAM_CPU += 4*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double) / 1e6;
		cudaMalloc((void**) &Dev_particles_pos, 2*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double));
		cudaMalloc((void**) &Dev_particles_vel, 2*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double));
		mb_used_RAM_GPU += 4*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double) / 1e6;

		// initialize randomizer
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		// set seed
		curandSetPseudoRandomGeneratorSeed(prng, SettingsMain.getParticlesSeed());
		// create initial positions from random distribution
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles);

		// Particles position initialization, make here to be absolutely sure to have the same positions
		// project 0-1 onto particle frame
		k_rescale<<<particle_block, particle_thread>>>(SettingsMain.getParticlesNum(),
				SettingsMain.getParticlesCenterX(), SettingsMain.getParticlesCenterY(), SettingsMain.getParticlesWidthX(), SettingsMain.getParticlesWidthY(),
				Dev_particles_pos, LX, LY);

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p+=1)
			cudaMemcpy(Dev_particles_pos + 2*Nb_particles*index_tau_p, Dev_particles_pos, 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		// particles where every time step the position will be saved
		if (SettingsMain.getParticlesSnapshotsPerSec() > 0) {

			if (SettingsMain.getSaveFineParticles()) {
				SettingsMain.setParticlesFineNum(std::min(SettingsMain.getParticlesFineNum(), SettingsMain.getParticlesNum()));

				// how many fine particles do we estimate max
				particles_fine_max = 2.0/(double)SettingsMain.getParticlesSnapshotsPerSec()/dt;  // normal case
				// allocate ram memory for this
				Host_particles_fine_pos = new double[(int)(2*particles_fine_max*SettingsMain.getParticlesTauNum()*SettingsMain.getParticlesFineNum())];
			}
		}

		create_particle_directory_structure(SettingsMain);

		if (SettingsMain.getVerbose() >= 1) {
			message = "Number of particles = " + to_str(Nb_particles);
			std::cout<<message+"\n"; logger.push(message);
			if (SettingsMain.getParticlesSnapshotsPerSec() > 0 && SettingsMain.getSaveFineParticles()) {
				message = "Number of fine particles = " + to_str(SettingsMain.getParticlesFineNum());
				std::cout<<message+"\n"; logger.push(message);
			}
		}
	}


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	int count_mesure = 0; int count_mesure_sample = 0;
	int mes_size = 2*SettingsMain.getConvInitFinal();  // initial and last step if computed
	if (SettingsMain.getConvSnapshotsPerSec() > 0) {
		mes_size += (int)(tf*SettingsMain.getConvSnapshotsPerSec());  // add all intermediate targets
	}
    double *Mesure, *Mesure_fine, *Mesure_sample;
	cudaMallocManaged(&Mesure, (3*mes_size+1)*sizeof(double));  // + 1 because i dont know what it does with 0
	cudaMallocManaged(&Mesure_fine, (3*mes_size+1)*sizeof(double));

	int mes_sample_size = SettingsMain.getSampleSaveInitial() + SettingsMain.getSampleSaveFinal();  // initial and last step if computed
	if (SettingsMain.getSampleSnapshotsPerSec() > 0) {
		mes_sample_size += (int)(tf*SettingsMain.getSampleSnapshotsPerSec());  // add all intermediate targets
	}
	if (SettingsMain.getSampleOnGrid()) {
		cudaMallocManaged(&Mesure_sample, (3*mes_sample_size+1)*sizeof(double));
	}

    double* incomp_error = new double[iterMax];  // save incompressibility error for investigations
    double* map_gap = new double[iterMax];  // save map gaps for investigations
    double* map_ctr = new double[iterMax];  // save map counter for investigations
    double* time_values = new double[iterMax+2];  // save timing for investigations
    mb_used_RAM_CPU += 6*iterMax*sizeof(double) / 1e6;  // +2 from t and dt vector

    // Laplacian
	/*
    double *Host_lap_fine, *Dev_lap_fine_real;
    cufftDoubleComplex *Dev_lap_fine_complex, *Dev_lap_fine_hat;

    Host_lap_fine = new double[Grid_fine.N];

    cudaMalloc((void**)&Dev_lap_fine_real, Grid_fine.sizeNReal);
    cudaMalloc((void**)&Dev_lap_fine_complex, Grid_fine.sizeNComplex);
    cudaMalloc((void**)&Dev_lap_fine_hat, Grid_fine.sizeNComplex);
	*/
	
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
	if (SettingsMain.getSampleOnGrid()) size_max_temp_2 = std::max(size_max_temp_2, 3*Grid_sample.sizeNReal + Grid_sample.sizeNfft);
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
		
	//initialization of flow map as normal grid
	k_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse);

	//setting initial conditions for vorticity by translating with initial grid
	translate_initial_condition_through_map_stack(Grid_fine, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
			cufft_plan_fine_D2Z, cufft_plan_fine_Z2D, Dev_Temp_C1,
			Dev_W_H_initial, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete());

	// compute first psi, stream hermite from vorticity
	evaluate_stream_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
			Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
			Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());


	/*
	 * Initialization of previous velocities for lagrange interpolation
	 * Either:  Compute backwards EulerExp for all needed velocities
	 * Or:		Increase order by back-and-forth computation until previous velocities are of desired order
	 */
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

				// map advection
				advect_using_stream_hermite(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
						(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N, Dev_Psi_real, t_init, dt_init, 0);
				cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)(Dev_Temp_C1), 			   		 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
				cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
				cudaDeviceSynchronize();

				// test: incompressibility error
				double incomp_init = incompressibility_check(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Grid_fine, Grid_coarse);

				//copy Psi to previous locations, from oldest-1 upwards
				for (int i_lagrange = 1; i_lagrange <= i_order-(i_order==lagrange_order_init); i_lagrange++) {
					cudaMemcpy(Dev_Psi_real + 4*Grid_psi.N*(i_order-(i_order==lagrange_order_init)-i_lagrange+1),
							   Dev_Psi_real + 4*Grid_psi.N*(i_order-(i_order==lagrange_order_init)-i_lagrange), 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
				}
				// compute stream hermite from vorticity for computed time step
				evaluate_stream_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
						Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
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


	// save function to save variables, combined so we always save in the same way and location
    // use Dev_Hat_fine for W_fine, this works because just at the end of conservation it is overwritten
	if (SettingsMain.getSaveInitial() || SettingsMain.getConvInitFinal()) {
		// fine vorticity disabled, as this needs another Grid_fine.sizeNReal in temp buffer, need to resolve later
//		apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
//				(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

		if (SettingsMain.getSaveInitial()) {
			writeTimeStep(SettingsMain, "0", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real,
					Dev_ChiX, Dev_ChiY, Grid_fine, Grid_coarse, Grid_psi);
		}

		// compute conservation if wanted, not connected to normal saving to reduce data
		if (SettingsMain.getConvInitFinal()) {
			compute_conservation_targets(Grid_fine, Grid_coarse, Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1,
					cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D,
					Dev_Temp_C1, Mesure, Mesure_fine, count_mesure);
			if (SettingsMain.getVerbose() >= 3) {
				message = "Coarse Cons : Energ = " + to_str(Mesure[3*count_mesure], 8)
						+    " \t Enstr = " + to_str(Mesure[3*count_mesure+1], 8)
						+ " \t Palinstr = " + to_str(Mesure[3*count_mesure+2], 8);
				std::cout<<message+"\n"; logger.push(message);
			}
			count_mesure++;
		}

	}
	// sample if wanted
	if (SettingsMain.getSampleOnGrid() && SettingsMain.getSampleSaveInitial()) {
		sample_compute_and_write(Map_Stack, Grid_sample, Grid_discrete, Host_save, Dev_Temp_2,
				cufft_plan_sample_D2Z, cufft_plan_sample_Z2D, Dev_Temp_C1,
				Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain, "0",
				Mesure_sample, count_mesure_sample);
		if (SettingsMain.getVerbose() >= 3) {
			message = "Sample Cons : Energ = " + to_str(Mesure_sample[3*count_mesure_sample], 8)
					+    " \t Enstr = " + to_str(Mesure_sample[3*count_mesure_sample+1], 8)
					+ " \t Palinstr = " + to_str(Mesure_sample[3*count_mesure_sample+2], 8);
			std::cout<<message+"\n"; logger.push(message);
		}
		count_mesure_sample++;
	}

    cudaDeviceSynchronize();

    // now lets get the particles in
    if (SettingsMain.getParticles()) {

		// velocity initialization for inertial particles
		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p++){
			Particle_advect_inertia_init<<<particle_block, particle_thread>>>(Nb_particles, dt,
					Dev_particles_pos + 2*Nb_particles*index_tau_p, Dev_particles_vel + 2*Nb_particles*index_tau_p,
					Dev_Psi_real, Grid_psi);
		}

		if (SettingsMain.getParticlesSaveInitial()) {
			writeParticles(SettingsMain, "0", Host_particles_pos, Dev_particles_pos);
		}
	}

	// zoom if wanted, has to be done after particle initialization, maybe a bit useless at first instance
	if (SettingsMain.getZoom() && SettingsMain.getZoomSaveInitial()) {
		Zoom(SettingsMain, Map_Stack, Grid_zoom, Grid_psi, Grid_discrete, Dev_ChiX, Dev_ChiY,
				(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, Dev_Psi_real,
				Host_particles_pos, Dev_particles_pos, Host_save, "0");
	}


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


	// first timing save before loop - this is the initialization time
	{
		auto step = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
		time_values[loop_ctr] = diff; // loop_ctr was already increased
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

	while(tf - t_vec[loop_ctr + SettingsMain.getLagrangeOrder()-1] > dt*1e-5 && loop_ctr < iterMax)
	{
		/*
		 * Timestep initialization:
		 *  - avoid overstepping final time
		 *  - avoid overstepping targets
		 *  - save dt and t into vectors for lagrange interpolation
		 */
		int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;
		double dt_now = dt;  // compute current timestep size

		// avoid overstepping final time
		if(t_vec[loop_ctr_l] + dt > tf) dt_now = tf - t_vec[loop_ctr_l];
		// avoid overstepping specific targets
		double targets[5] = {SettingsMain.getSnapshotsPerSec(),
						  SettingsMain.getConvSnapshotsPerSec(),
						  SettingsMain.getSampleOnGrid()*SettingsMain.getSampleSnapshotsPerSec(),
						  SettingsMain.getZoom()*SettingsMain.getZoomSnapshotsPerSec(),
						  SettingsMain.getParticles()*SettingsMain.getParticlesSnapshotsPerSec(),
//						  4/PI  // enforce one step change to test convergence behaviour
		};
		for ( const auto &i_target : targets) {
			if (i_target > 0) {
				double target_now = 1.0/(double)i_target;
				if(fmod(t_vec[loop_ctr_l]+dt*1e-5, target_now) > fmod(t_vec[loop_ctr_l] + dt*(1+1e-5), target_now)) {
					dt_now = fmin(dt_now, target_now - fmod(t_vec[loop_ctr_l], target_now));
				}
			}
		}

		// set new dt into time vectors for lagrange interpolation
		t_vec[loop_ctr_l + 1] = t_vec[loop_ctr_l] + dt_now;
		dt_vec[loop_ctr_l + 1] = dt_now;

		/*
		 * Map advection
		 *  - Velocity is already intialized, so we can safely do that here
		 */
	    // by grid itself - only useful for very large grid sizes (has to be investigated), not recommended
	    if (SettingsMain.getMapUpdateGrid()) {
	    advect_using_stream_hermite_grid(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
	    		(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1 + 4*Grid_coarse.N, Dev_Psi_real, t_vec, dt_vec, loop_ctr);
	    }
	    // by footpoints
	    else {
		    advect_using_stream_hermite(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
		    		(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N, Dev_Psi_real, t_vec, dt_vec, loop_ctr);
			cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)(Dev_Temp_C1), 			   		 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
			cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)(Dev_Temp_C1) + 4*Grid_coarse.N, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
	    }
	    cudaDeviceSynchronize();


		/*******************************************************************
		*							 Remapping							   *
		*				Computing incompressibility
		*				Checking against threshold
		*				If exceeded: Safe map on CPU RAM
		*							 Initialize variables
		*******************************************************************/
		incomp_error[loop_ctr] = incompressibility_check(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Grid_fine, Grid_coarse);
//		incomp_error[loop_ctr] = incompressibility_check(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Grid_coarse, Grid_coarse);

		//resetting map and adding to stack
		if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) {

			//		if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) {
			if(Map_Stack.map_stack_ctr > Map_Stack.cpu_map_num*Map_Stack.Nb_array_RAM)
			{
				if (SettingsMain.getVerbose() >= 0) {
					message = "Stack Saturated : Exiting"; std::cout<<message+"\n"; logger.push(message);
				}
				break;
			}
			
			if (SettingsMain.getVerbose() >= 2) {
				message = "Refining Map : Step = " + to_str(loop_ctr) + " \t Maps = " + to_str(Map_Stack.map_stack_ctr) + " ; "
						+ to_str(Map_Stack.map_stack_ctr/Map_Stack.cpu_map_num) + " \t Gap = " + to_str(loop_ctr - old_ctr);
				std::cout<<message+"\n"; logger.push(message);
			}
			old_ctr = loop_ctr;
			
			//adjusting initial conditions, compute vorticity hermite
			translate_initial_condition_through_map_stack(Grid_fine, Grid_discrete, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
					cufft_plan_fine_D2Z, cufft_plan_fine_Z2D, Dev_Temp_C1,
					Dev_W_H_initial, SettingsMain.getInitialConditionNum(), SettingsMain.getInitialDiscrete());
			
			if ((Map_Stack.map_stack_ctr%Map_Stack.cpu_map_num) == 0 && SettingsMain.getVerbose() >= 2){
				message = "Starting to use map stack array number " + to_str(Map_Stack.map_stack_ctr/Map_Stack.cpu_map_num);
				std::cout<<message+"\n"; logger.push(message);
			}

			Map_Stack.copy_map_to_host(Dev_ChiX, Dev_ChiY);
			
			//resetting map
			k_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse);
		}
		// save map invetigative details
		map_gap[loop_ctr] = loop_ctr - old_ctr;
		map_ctr[loop_ctr] = Map_Stack.map_stack_ctr;


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
		evaluate_stream_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY,
				Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufft_plan_coarse_Z2D, cufft_plan_psi_Z2D, cufft_plan_vort_D2Z,
				Dev_Temp_C1, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

		/*
		 * Particles advection after velocity update to profit from nice avaiable accelerated schemes
		 */
		// Particles advection after velocity update to profit from nice avaiable schemes
	    if (SettingsMain.getParticles()) {

	    	particles_advect(SettingsMain, Grid_psi, Dev_particles_pos, Dev_particles_vel, Dev_Psi_real,
	    			t_vec, dt_vec, loop_ctr, particle_block, particle_thread);

			// copy fine particle positions
			if (SettingsMain.getParticlesSnapshotsPerSec() > 0 && SettingsMain.getSaveFineParticles()) {
				for(int i_tau_p = 1; i_tau_p < SettingsMain.getParticlesTauNum(); i_tau_p++){
					cudaMemcpy(Host_particles_fine_pos + 2*((loop_ctr-particles_fine_old)*SettingsMain.getParticlesTauNum()+i_tau_p)*SettingsMain.getParticlesFineNum(),
							Dev_particles_pos + 2*SettingsMain.getParticles()*i_tau_p, 2*SettingsMain.getParticlesTauNum()*SettingsMain.getParticlesFineNum()*sizeof(double), cudaMemcpyDeviceToHost);
				}
			}
	    }


			/*******************************************************************
			*							 Save snap shot						   *
			*				Normal variables in their respective grid
			*				Variables on sampled grid if wanted
			*				Particles together with fine particles
			*******************************************************************/
	    if (SettingsMain.getSnapshotsPerSec() > 0) {
			if( fmod(t_vec[loop_ctr_l+1]+dt*1e-5, 1.0/(double)SettingsMain.getSnapshotsPerSec()) < dt_vec[loop_ctr_l+1] ) {
				if (SettingsMain.getVerbose() >= 2) {
					message = "Saving data : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
					std::cout<<message+"\n"; logger.push(message);
				}

				// save function to save variables, combined so we always save in the same way and location
				// fine vorticity disabled, as this needs another Grid_fine.sizeNReal in temp buffer, need to resolve later
//				apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
//						(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

				writeTimeStep(SettingsMain, to_str(t_vec[loop_ctr_l+1]), Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, Grid_fine, Grid_coarse, Grid_psi);
			}
	    }

	    if (SettingsMain.getConvSnapshotsPerSec() > 0) {
			if( fmod(t_vec[loop_ctr_l+1]+dt*1e-5, 1.0/(double)SettingsMain.getConvSnapshotsPerSec()) < dt_vec[loop_ctr_l+1] ) {
				// fine vorticity disabled, as this needs another Grid_fine.sizeNReal in temp buffer, need to resolve later
//				apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
//						(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

				// compute conservation
				compute_conservation_targets(Grid_fine, Grid_coarse, Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1,
						cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D,
						Dev_Temp_C1, Mesure, Mesure_fine, count_mesure);
				if (SettingsMain.getVerbose() >= 3) {
					message = "Coarse Cons : Energ = " + to_str(Mesure[3*count_mesure], 8)
							+    " \t Enstr = " + to_str(Mesure[3*count_mesure+1], 8)
							+ " \t Palinstr = " + to_str(Mesure[3*count_mesure+2], 8);
					std::cout<<message+"\n"; logger.push(message);
				}
				count_mesure++;
			}
	    }
	    if (SettingsMain.getSampleOnGrid()*SettingsMain.getSampleSnapshotsPerSec() > 0) {
	    	if( fmod(t_vec[loop_ctr_l+1]+dt*1e-5, 1.0/(double)SettingsMain.getSampleSnapshotsPerSec()) < dt_vec[loop_ctr_l+1] ) {
				if (SettingsMain.getVerbose() >= 2) {
					message = "Saving sample data : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
					std::cout<<message+"\n"; logger.push(message);
				}
				sample_compute_and_write(Map_Stack, Grid_sample, Grid_discrete, Host_save, Dev_Temp_2,
						cufft_plan_sample_D2Z, cufft_plan_sample_Z2D, Dev_Temp_C1,
						Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain, to_str(t_vec[loop_ctr_l+1]),
						Mesure_sample, count_mesure_sample);
				if (SettingsMain.getVerbose() >= 3) {
					message = "Sample Cons : Energ = " + to_str(Mesure_sample[3*count_mesure_sample], 8)
							+    " \t Enstr = " + to_str(Mesure_sample[3*count_mesure_sample+1], 8)
							+ " \t Palinstr = " + to_str(Mesure_sample[3*count_mesure_sample+2], 8);
					std::cout<<message+"\n"; logger.push(message);
				}
				count_mesure_sample++;
	    	}
		}

		if (SettingsMain.getParticles()*SettingsMain.getParticlesSnapshotsPerSec() > 0) {
			if( fmod(t_vec[loop_ctr_l+1]+dt*1e-5, 1.0/(double)SettingsMain.getParticlesSnapshotsPerSec()) < dt_vec[loop_ctr_l+1] ) {
				if (SettingsMain.getVerbose() >= 2) {
					message = "Saving particles : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
					std::cout<<message+"\n"; logger.push(message);
				}

				// save particle positions
				if (SettingsMain.getParticles()) {
					writeParticles(SettingsMain, to_str(t_vec[loop_ctr_l+1]), Host_particles_pos, Dev_particles_pos);

					// if wanted, save fine particles, 1 file for all positions
					if (SettingsMain.getSaveFineParticles()) {
						writeFineParticles(SettingsMain, to_str(t_vec[loop_ctr_l+1]), Host_particles_fine_pos, (loop_ctr - particles_fine_old + 1)*SettingsMain.getParticlesTauNum()*SettingsMain.getParticlesFineNum());
						particles_fine_old = loop_ctr+1;
					}
				}
			}
		}
		
		// zoom if wanted, has to be done after particle initialization, maybe a bit useless at first instance
		if (SettingsMain.getZoom()*SettingsMain.getZoomSnapshotsPerSec() > 0) {
			if( fmod(t_vec[loop_ctr_l+1]+dt*1e-5, 1.0/(double)SettingsMain.getZoomSnapshotsPerSec()) < dt_vec[loop_ctr_l+1] ) {
				if (SettingsMain.getVerbose() >= 2) {
					message = "Saving Zoom : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
					std::cout<<message+"\n"; logger.push(message);
				}

				Zoom(SettingsMain, Map_Stack, Grid_zoom, Grid_psi, Grid_discrete, Dev_ChiX, Dev_ChiY,
						(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, Dev_Psi_real,
						Host_particles_pos, Dev_particles_pos, Host_save, to_str(t_vec[loop_ctr_l+1]));
			}
		}

		/*
		 * Some small things at the end of the loop
		 *  - check for errors
		 *  - spit out some verbose information
		 *  - safe timing of loop
		 */

		//loop counters are increased at the end
		loop_ctr ++;

		int error = cudaGetLastError();
		if(error != 0)
		{
			if (SettingsMain.getVerbose() >= 0) {
				message = "Exited early : Last Cuda Error = " + to_str(error); std::cout<<message+"\n"; logger.push(message);
			}
			exit(0);
			break;
		}

		// save timing at last part of a step to take everything into account
		{
			auto step = std::chrono::high_resolution_clock::now();
			double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
			time_values[loop_ctr] = diff; // loop_ctr was already increased but first entry is init time
		}
		if (SettingsMain.getVerbose() >= 2) {
			message = "Step = " + to_str(loop_ctr)
					+ " \t S-Time = " + to_str(t_vec[loop_ctr_l+1]) + "/" + to_str(tf)
					+ " \t IncompErr = " + to_str(incomp_error[loop_ctr-1])
					+ " \t C-Time = " + format_duration(time_values[loop_ctr]);
			std::cout<<message+"\n"; logger.push(message);
		}

	}
	
	// introduce part to console
	if (SettingsMain.getVerbose() >= 2) {
		message = "Simulation loop finished"
				  " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}

	
	// hackery: new loop for particles to test convergence without map influence
	if (SettingsMain.getParticlesSteps() != -1) {

		// copy velocity to old values to be uniform and constant in time
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpyAsync(Dev_Psi_real + 4*Grid_psi.N*i_lagrange, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[i_lagrange]);
		}
		// initialize particles again
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles);

		// project 0-1 onto particle frame
		k_rescale<<<particle_block, particle_thread>>>(SettingsMain.getParticlesNum(),
				SettingsMain.getParticlesCenterX(), SettingsMain.getParticlesCenterY(), SettingsMain.getParticlesWidthX(), SettingsMain.getParticlesWidthY(),
				Dev_particles_pos, LX, LY);

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p+=1)
			cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p+=1){
			Particle_advect_inertia_init<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p],
					&Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Grid_psi);
		}
		// save inital position to check if they are qual
		writeParticles(SettingsMain, "C0", Host_particles_pos, Dev_particles_pos);

		auto begin_P = std::chrono::high_resolution_clock::now();
		if (SettingsMain.getVerbose() >= 2) {
			message = "Starting extra particle loop : Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(begin_P - begin).count()/1e6);
			std::cout<<message+"\n"; logger.push(message);
		}

		// main loop until time 1
		double dt_p = 1/(double)SettingsMain.getParticlesSteps();
		for (int loop_ctr_p = 0; loop_ctr_p < SettingsMain.getParticlesSteps(); ++loop_ctr_p) {
			// particles advection
	    	particles_advect(SettingsMain, Grid_psi, Dev_particles_pos, Dev_particles_vel, Dev_Psi_real,
	    			t_vec, dt_vec, loop_ctr, particle_block, particle_thread);
		}

		// force synchronize after loop to wait until everything is finished
		cudaDeviceSynchronize();

		auto end_P = std::chrono::high_resolution_clock::now();
		if (SettingsMain.getVerbose() >= 2) {
			message = "Finished extra particle loop : Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin).count()/1e6);
			std::cout<<message+"\n"; logger.push(message);
		}

		double time_p[1] = { std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin_P).count()/1e6 };
		writeAllRealToBinaryFile(1, time_p, SettingsMain, "/P_time");

		// save final position
		writeParticles(SettingsMain, "C1", Host_particles_pos, Dev_particles_pos);
	}

	
	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/

	// save function to save variables, combined so we always save in the same way and location, last step so we can actually modify the condition
	if (SettingsMain.getSaveFinal() || SettingsMain.getConvInitFinal()) {
		// fine vorticity disabled, as this needs another Grid_fine.sizeNReal in temp buffer, need to resolve later
//		apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
//				(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

		if (SettingsMain.getSaveFinal()) {
			writeTimeStep(SettingsMain, "final", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real,
					Dev_ChiX, Dev_ChiY, Grid_fine, Grid_coarse, Grid_psi);
		}

		// compute conservation if wanted, not connected to normal saving to reduce data
		if (SettingsMain.getConvInitFinal()) {
			compute_conservation_targets(Grid_fine, Grid_coarse, Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1,
					cufft_plan_coarse_D2Z, cufft_plan_coarse_Z2D, cufft_plan_fine_D2Z, cufft_plan_fine_Z2D,
					Dev_Temp_C1, Mesure, Mesure_fine, count_mesure);
			if (SettingsMain.getVerbose() >= 3) {
				message = "Coarse Cons : Energ = " + to_str(Mesure[3*count_mesure], 8)
						+    " \t Enstr = " + to_str(Mesure[3*count_mesure+1], 8)
						+ " \t Palinstr = " + to_str(Mesure[3*count_mesure+2], 8);
				std::cout<<message+"\n"; logger.push(message);
			}
			count_mesure++;
		}

	}

	// sample if wanted
	if (SettingsMain.getSampleOnGrid() && SettingsMain.getSampleSaveFinal()) {
		sample_compute_and_write(Map_Stack, Grid_sample, Grid_discrete, Host_save, Dev_Temp_2,
				cufft_plan_sample_D2Z, cufft_plan_sample_Z2D, Dev_Temp_C1,
				Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain, "final",
				Mesure_sample, count_mesure_sample);
		if (SettingsMain.getVerbose() >= 3) {
			message = "Sample Cons : Energ = " + to_str(Mesure_sample[3*count_mesure_sample], 8)
					+    " \t Enstr = " + to_str(Mesure_sample[3*count_mesure_sample+1], 8)
					+ " \t Palinstr = " + to_str(Mesure_sample[3*count_mesure_sample+2], 8);
			std::cout<<message+"\n"; logger.push(message);
		}
		count_mesure_sample++;
	}
	if (SettingsMain.getParticles() && SettingsMain.getParticlesSaveFinal()) {
		writeParticles(SettingsMain, "final", Host_particles_pos, Dev_particles_pos);
	}

	// zoom if wanted
	if (SettingsMain.getZoom() && SettingsMain.getZoomSaveFinal()) {
		Zoom(SettingsMain, Map_Stack, Grid_zoom, Grid_psi, Grid_discrete, Dev_ChiX, Dev_ChiY,
				(cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial, Dev_Psi_real,
				Host_particles_pos, Dev_particles_pos, Host_save, "final");
	}

	// save all conservation data, little switch in case we do not save anything
	if (mes_size > 0) {
		writeAllRealToBinaryFile(3*mes_size, Mesure, SettingsMain, "/Monitoring_data/Mesure");
		writeAllRealToBinaryFile(3*mes_size, Mesure_fine, SettingsMain, "/Monitoring_data/Mesure_fine");
	}
	if (SettingsMain.getSampleOnGrid() && mes_sample_size > 0) {
		writeAllRealToBinaryFile(3*mes_sample_size, Mesure_sample, SettingsMain, "/Monitoring_data/Mesure_"+to_str(Grid_sample.NX));
	}

	// save timings
	writeAllRealToBinaryFile(loop_ctr, t_vec+SettingsMain.getLagrangeOrder(), SettingsMain, "/Monitoring_data/Timesteps");
    // save imcomp error
	writeAllRealToBinaryFile(loop_ctr, incomp_error, SettingsMain, "/Monitoring_data/Incompressibility_check");
	// save map monitorings
	writeAllRealToBinaryFile(loop_ctr, map_gap, SettingsMain, "/Monitoring_data/Map_gaps");
	writeAllRealToBinaryFile(loop_ctr, map_ctr, SettingsMain, "/Monitoring_data/Map_counter");

	
	// save map stack if wanted
	if (SettingsMain.getSaveMapStack()) {
		// last map should be included too, so we have a full stack until this time
		if (Map_Stack.map_stack_ctr < Map_Stack.cpu_map_num*Map_Stack.Nb_array_RAM) {
			Map_Stack.copy_map_to_host(Dev_ChiX, Dev_ChiY);
		}
		if (SettingsMain.getVerbose() >= 2) {
			message = "Saving MapStack : Maps = " + to_str(Map_Stack.map_stack_ctr) + " \t Filesize = " + to_str(Map_Stack.map_stack_ctr*map_size) + "mb"; std::cout<<message+"\n"; logger.push(message);
		}
		writeMapStack(SettingsMain, Map_Stack);
	}

	
	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/

	cudaFree(Dev_W_H_initial);
	
	// Trash variable
	cudaFree(Dev_Temp_C1);
	if (size_max_temp_2 != 0) cudaFree(Dev_Temp_2);
	delete [] Host_save;
	
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
	if (SettingsMain.getSampleOnGrid()) {
		cufftDestroy(cufft_plan_sample_D2Z); cufftDestroy(cufft_plan_sample_Z2D);
	}
	cudaFree(fft_work_area);

	if (SettingsMain.getParticles()) {
	// Particles variables
	    delete [] Host_particles_pos;
	    delete [] Host_particles_vel;
	    delete [] Host_particles_fine_pos;
	    cudaFree(Dev_particles_pos);
        cudaFree(Dev_particles_vel);
	}

    cudaFree(Mesure);
    cudaFree(Mesure_fine);
	if (SettingsMain.getSampleOnGrid()) {
		cudaFree(Mesure_sample);
	}

	// delete monitoring values
	delete []     incomp_error, map_gap, map_ctr, time_values, t_vec, dt_vec;

	// save timing at last part of a step to take everything into account
    {
		auto step = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
		time_values[loop_ctr+1] = diff;
    }
    // save timing to file
	writeAllRealToBinaryFile(loop_ctr+2, time_values, SettingsMain, "/Monitoring_data/Timing_Values");

	// introduce part to console
	if (SettingsMain.getVerbose() >= 1) {
		message = "Finished simulation"
				  " \t Last Cuda Error = " + to_str(cudaGetErrorName(cudaGetLastError()))
				+ " \t C-Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		std::cout<<"\n"+message+"\n\n"; logger.push(message);
	}
}



