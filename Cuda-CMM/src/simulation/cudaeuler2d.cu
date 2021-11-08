#include "cudaeuler2d.h"

#include <curand.h>
#include <curand_kernel.h>

#include "../numerical/cmm-particles.h"
#include "../simulation/cudamesure2d.h"
#include "../simulation/cudasimulation2d.h"
#include "../simulation/cmm-fft.h"

#include "../ui/cmm-io.h"

#include <unistd.h>
#include <chrono>

// parallel reduce
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


void cuda_euler_2d(SettingsCMM SettingsMain)
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
	int iterMax;														// time - maximum iteration count

	string workspace = SettingsMain.getWorkspace();						// folder where we work in
	string sim_name = SettingsMain.getSimName();						// name of the simulation
	string initial_condition = SettingsMain.getInitialCondition();		// name of the initial condition
	int snapshots_per_second = SettingsMain.getSnapshotsPerSec();		// saves per second
	int save_buffer_count;												// iterations after which files should be saved

	double mb_used_RAM_GPU;  // count memory usage by variables in mbyte

	// compute dt, for grid settings we have to use max in case we want to differ NX and NY
	if (SettingsMain.getSetDtBySteps()) dt = 1.0 / steps_per_sec;  // direct setting of timestep
	else dt = 1.0 / std::max(NX_coarse, NY_coarse) / grid_by_time;  // setting of timestep by grid and factor
	
	// reset lagrange order
	if (SettingsMain.getLagrangeOverride() != -1) SettingsMain.setLagrangeOrder(SettingsMain.getLagrangeOverride());
	
	//shared parameters
	iterMax = ceil(tf / dt);
	if (snapshots_per_second > 0) {
		save_buffer_count = (int)(1.0/(double)snapshots_per_second/dt);  // normal case
	}
	else {
		save_buffer_count = INT_MIN;  // disable saving
	}

	double map_size = 8*NX_coarse*NY_coarse*sizeof(double) / 1e6;  // size of one mapping
	int Nb_array_RAM = 4;  // fixed for four different stacks
	int cpu_map_num = int(double(SettingsMain.getMemRamCpuRemaps())/map_size/double(Nb_array_RAM));  // define how many more remappings we can save on CPU than on GPU
	
	// build file name together
	string file_name = sim_name + "_" + initial_condition + "_C" + to_str(NX_coarse) + "_F" + to_str(NX_fine) + "_t" + to_str(1.0/dt) + "_T" + to_str(tf);
	SettingsMain.setFileName(file_name);

	create_directory_structure(SettingsMain, dt, save_buffer_count, iterMax);
    Logger logger(SettingsMain);

    string message;  // string to be used for console output
    if (SettingsMain.getVerbose() >= 1) {
		message = "Initial condition = " + initial_condition; cout<<message+"\n"; logger.push(message);
		message = "Iter max = " + to_str(iterMax); cout<<message+"\n"; logger.push(message);
		message = "Save buffer count = " + to_str(save_buffer_count); cout<<message+"\n"; logger.push(message);
		message = "Map stack length on CPU = " + to_str(cpu_map_num); cout<<message+"\n"; logger.push(message);
		message = "Map stack length total on CPU = " + to_str(cpu_map_num * Nb_array_RAM); cout<<message+"\n"; logger.push(message);
		message = "Name of simulation = " + SettingsMain.getFileName(); cout<<message+"\n"; logger.push(message);
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
	
	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 	we never run ffts in parallel, so we can reuse the temp space  *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufftPlan_coarse, cufftPlan_fine, cufftPlan_psi, cufftPlan_vort, cufftPlan_sample, cufftPlan_zoom;
	// preinitialize handles
	cufftCreate(&cufftPlan_coarse); cufftCreate(&cufftPlan_fine);
	cufftCreate(&cufftPlan_psi); cufftCreate(&cufftPlan_vort);
	cufftCreate(&cufftPlan_sample); cufftCreate(&cufftPlan_zoom);

	// disable auto workspace creation for fft plans
	cufftSetAutoAllocation(cufftPlan_coarse, 0); cufftSetAutoAllocation(cufftPlan_fine, 0);
	cufftSetAutoAllocation(cufftPlan_psi, 0); cufftSetAutoAllocation(cufftPlan_vort, 0);
	cufftSetAutoAllocation(cufftPlan_sample, 0); cufftSetAutoAllocation(cufftPlan_zoom, 0);

	// create plans and compute needed size of each plan
	size_t workSize[6];
	cufftMakePlan2d(cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z, &workSize[0]);
	cufftMakePlan2d(cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z, &workSize[1]);
	cufftMakePlan2d(cufftPlan_psi, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z, &workSize[2]);
	cufftMakePlan2d(cufftPlan_vort, Grid_vort.NX, Grid_vort.NY, CUFFT_Z2Z, &workSize[3]);
	if (SettingsMain.getSampleOnGrid()) {
		cufftMakePlan2d(cufftPlan_sample, Grid_sample.NX, Grid_sample.NY, CUFFT_Z2Z, &workSize[4]);
	}
	if (SettingsMain.getZoom() && SettingsMain.getZoomSavePsi()) {
		cufftMakePlan2d(cufftPlan_zoom, Grid_zoom.NX, Grid_zoom.NY, CUFFT_Z2Z, &workSize[5]);
	}

    // allocate memory to new workarea for cufft plans with maximum size
	size_t size_max_fft = std::max(workSize[0], workSize[1]);
	size_max_fft = std::max(size_max_fft, workSize[2]);
	size_max_fft = std::max(size_max_fft, workSize[3]);
	if (SettingsMain.getSampleOnGrid()) {
		size_max_fft = std::max(size_max_fft, workSize[4]);
	}
	if (SettingsMain.getZoom() && SettingsMain.getZoomSavePsi()) {
		size_max_fft = std::max(size_max_fft, workSize[5]);
	}
	void *fft_work_area;
	cudaMalloc(&fft_work_area, size_max_fft);
	mb_used_RAM_GPU = size_max_fft / 1e6;

	// set new workarea to plans
	cufftSetWorkArea(cufftPlan_coarse, fft_work_area); cufftSetWorkArea(cufftPlan_fine, fft_work_area);
	cufftSetWorkArea(cufftPlan_psi, fft_work_area); cufftSetWorkArea(cufftPlan_vort, fft_work_area);
	if (SettingsMain.getSampleOnGrid()) {
		cufftSetWorkArea(cufftPlan_sample, fft_work_area);
	}
	if (SettingsMain.getZoom() && SettingsMain.getZoomSavePsi()) {
		cufftSetWorkArea(cufftPlan_zoom, fft_work_area);
	}

////	size_t workSize;
//	cufftPlan2d(&cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z);
////	cufftGetSize(cufftPlan_coarse, &workSize);
////	mb_used_RAM_GPU += workSize / 1e6;
//	cufftPlan2d(&cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z);
////	cufftGetSize(cufftPlan_fine, &workSize);
////	mb_used_RAM_GPU += workSize / 1e6;
//	cufftPlan2d(&cufftPlan_psi, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z);
////	cufftGetSize(cufftPlan_psi, &workSize);
////	mb_used_RAM_GPU += workSize / 1e6;
//	if (SettingsMain.getSampleOnGrid()) {
//		cufftPlan2d(&cufftPlan_sample, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z);
////		cufftGetSize(cufftPlan_sample, &workSize);
////		mb_used_RAM_GPU += workSize / 1e6;
//	}
	
	
	/*******************************************************************
	*							Trash variable
	*          Used for temporary copying and storing of data
	*                       in various locations
	*  casting (cufftDoubleReal*) makes them usable for double arrays
	*******************************************************************/
	
	// set after largest grid, checks are for failsave in case one chooses weird grid
	long int size_max_c = std::max(Grid_fine.sizeNComplex, 4*Grid_coarse.sizeNReal);
	size_max_c = std::max(size_max_c, Grid_psi.sizeNComplex);
	size_max_c = std::max(size_max_c, Grid_vort.sizeNReal);
	if (SettingsMain.getSampleOnGrid()) {
		size_max_c = std::max(size_max_c, Grid_sample.sizeNComplex);
	}
	// for now three thrash variables are needed for cufft with hermites
	cufftDoubleComplex *Dev_Temp_C1, *Dev_Temp_C2;
	cudaMalloc((void**)&Dev_Temp_C1, size_max_c);
	cudaMalloc((void**)&Dev_Temp_C2, size_max_c);
	mb_used_RAM_GPU += 2*size_max_c / 1e6;
	
	// we actually only need one host array, as we always just copy from and to this and never really read files
	long int size_max_r = std::max(4*Grid_fine.N, 4*Grid_psi.N);
	size_max_r = std::max(size_max_r, 4*Grid_coarse.N);
	if (SettingsMain.getSampleOnGrid()) {
		size_max_r = std::max(size_max_r, 4*Grid_sample.N);
	}
	double *Host_save;
	Host_save = new double[size_max_r];

	
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
	if (SettingsMain.getVerbose() >= 2) {
		message = "Map Stack Initialized"; cout<<message+"\n"; logger.push(message);
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
	cudaMalloc((void**)&Dev_W_H_fine_real, 4*Grid_fine.sizeNReal);
	mb_used_RAM_GPU += 4*Grid_fine.sizeNReal / 1e6;
	
	
	/*******************************************************************
	*							DISCRET								   *
	*******************************************************************/
	
	double *Dev_W_H_initial;
	cudaMalloc((void**)&Dev_W_H_initial, sizeof(double));
	#ifdef DISCRET
		
		double *Host_W_initial, *Dev_W_H_initial;
		Host_W_initial = new double[Grid_fine.N];
		cudaMalloc((void**)&Dev_W_H_initial, 4*Grid_fine.sizeNReal);
		
		readRealToBinaryAnyFile(Grid_fine.N, Host_W_initial, "src/Initial_W_discret/file2D_" + to_str(NX_fine) + ".bin");
		
		cudaMemcpy(Dev_W_H_initial, Host_W_initial, Grid_fine.sizeNReal, cudaMemcpyHostToDevice);
		
		k_real_to_comp<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_W_H_initial, Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Temp_C1, CUFFT_FORWARD);
		k_normalize<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Temp_C1, Grid_fine.NX, Grid_fine.NY);
		
		// Hermite vorticity array : [vorticity, x-derivative, y-derivative, xy-derivative]
		fourier_hermite(Grid_fine, Dev_Temp_C1, Dev_W_H_initial, Dev_Temp_C2, cufftPlan_fine);
		delete [] Host_W_initial;
		
		message = cudaGetErrorName(cudaGetLastError()); cout<<message+"\n\n"; logger.push(message);
		
	#endif
	
	/*******************************************************************
	*							  Psi								   *
	* 	Psi is an array that contains Psi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative 							   *
	* 																   *
	*******************************************************************/
	
	//stream hermite on coarse computational grid, previous timesteps for lagrange interpolation included in array
	double *Dev_Psi_real;
	cudaMalloc((void**) &Dev_Psi_real, 4*SettingsMain.getLagrangeOrder()*Grid_psi.sizeNReal);
	mb_used_RAM_GPU += 4*SettingsMain.getLagrangeOrder()*Grid_psi.sizeNReal / 1e6;
	
	
	/*******************************************************************
	*							 Particles							   *
	*******************************************************************/
	// all variables have to be defined outside
	int Nb_particles, particle_thread, particle_block;
	double *Host_particles_pos ,*Dev_particles_pos;  // position of particles
	double *Host_particles_vel ,*Dev_particles_vel;  // velocity of particles
	curandGenerator_t prng;
	int particles_save_buffer_count, freq_fine_dt_particles, prod_fine_dt_particles;
	double *Dev_particles_fine_pos, *Host_particles_fine_pos;
	// now set the variables if we have particles
	if (SettingsMain.getParticles()) {
		// initialize all memory
		Nb_particles = SettingsMain.getParticlesNum();
		particle_thread =  256;  // threads for particles, seems good like that
		particle_block = Nb_particles / particle_thread + (Nb_particles < particle_thread);  // we need atleast 1 block
		Host_particles_pos = new double[2*Nb_particles*SettingsMain.getParticlesTauNum()];
		Host_particles_vel = new double[2*Nb_particles*SettingsMain.getParticlesTauNum()];
		cudaMalloc((void**) &Dev_particles_pos, 2*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double));
		cudaMalloc((void**) &Dev_particles_vel, 2*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double));
		mb_used_RAM_GPU += 4*Nb_particles*SettingsMain.getParticlesTauNum()*sizeof(double) / 1e6;

		// create initial positions from random distribution
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*SettingsMain.getParticlesTauNum());

		// Particles position initialization, make here to be absolutely sure to have the same positions
		Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, Dev_particles_pos);  // project 0-1 onto 0-LX

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p+=1)
			cudaMemcpy(Dev_particles_pos + 2*Nb_particles*index_tau_p, Dev_particles_pos, 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		// particles where every time step the position will be saved
		if (SettingsMain.getParticlesSnapshotsPerSec() > 0) {
			particles_save_buffer_count = (int)(1.0/(double)SettingsMain.getParticlesSnapshotsPerSec()/dt);  // normal case

			if (SettingsMain.getSaveFineParticles()) {
				SettingsMain.setParticlesFineNum(std::min(SettingsMain.getParticlesFineNum(), SettingsMain.getParticlesNum()));
				freq_fine_dt_particles = (int)(save_buffer_count); // save particles at every saving step
				prod_fine_dt_particles = SettingsMain.getParticlesFineNum() * freq_fine_dt_particles;

				Host_particles_fine_pos = new double[2*prod_fine_dt_particles];
				cudaMalloc((void**) &Dev_particles_fine_pos, 2*prod_fine_dt_particles*SettingsMain.getParticlesTauNum()*sizeof(double));
				mb_used_RAM_GPU += 2*prod_fine_dt_particles*SettingsMain.getParticlesTauNum()*sizeof(double) / 1e6;
			}
		}
		else particles_save_buffer_count = INT_MIN;  // disable saving

		create_particle_directory_structure(SettingsMain);

		if (SettingsMain.getVerbose() >= 1) {
			message = "Number of particles = " + to_str(Nb_particles);
			cout<<message+"\n"; logger.push(message);
			if (SettingsMain.getParticlesSnapshotsPerSec() > 0 && SettingsMain.getSaveFineParticles()) {
				message = "Number of fine particles = " + to_str(SettingsMain.getParticlesFineNum());
				cout<<message+"\n"; logger.push(message);
			}
		}
	}


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	int count_mesure = 0; int count_mesure_sample = 0;
	int mes_size = 2*SettingsMain.getConvInitFinal();  // initial and last step if computed
	if (snapshots_per_second > 0) {
		mes_size += (int)(tf*snapshots_per_second);  // add all intermediate targets
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

    double incomp_error [iterMax];  // save incompressibility error for investigations
    double map_gap [iterMax];  // save map gaps for investigations
    double map_ctr [iterMax];  // save map counter for investigations
    double time_values [iterMax+2];  // save timing for investigations

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
	
	double *Dev_Temp_3;
	long int size_max_temp_3 = 0;
	if (SettingsMain.getSampleOnGrid()) size_max_temp_3 = std::max(size_max_temp_3, 4*Grid_coarse.sizeNReal);
	if (SettingsMain.getZoom() && SettingsMain.getZoomSavePsi()) size_max_temp_3 = std::max(size_max_temp_3, 4*Grid_zoom.sizeNReal);
	// define third temporary variable as safe buffer to be used for sample and zoom
	if (size_max_temp_3 != 0) {
		cudaMalloc((void**)&Dev_Temp_3, size_max_temp_3);
		mb_used_RAM_GPU += size_max_temp_3 / 1e6;
	}

	int sample_save_buffer_count;

	if (SettingsMain.getSampleOnGrid() && SettingsMain.getSampleSnapshotsPerSec() > 0) {
		sample_save_buffer_count = 1.0/(double)SettingsMain.getSampleSnapshotsPerSec()/dt;  // normal case
	}
	else {
		sample_save_buffer_count = INT_MIN;  // disable saving
	}

	/*
	 * Zoom
	 */
	int zoom_save_buffer_count;
	// for psi we need a temp variable to safe in, this is defined in dev_temp_3
	if (SettingsMain.getZoom() && SettingsMain.getZoomSnapshotsPerSec() > 0) {
		zoom_save_buffer_count = 1.0/(double)SettingsMain.getZoomSnapshotsPerSec()/dt;  // normal case
	}
	else {
		zoom_save_buffer_count = INT_MIN;  // disable saving
	}


	// print estimated gpu memory usage in mb before initialization
	if (SettingsMain.getVerbose() >= 1) {
		message = "estimated GPU RAM usage in mb = " + to_str(mb_used_RAM_GPU);
		cout<<message+"\n"; logger.push(message);
	}



	/*******************************************************************
	*						   Initialization						   *
	*******************************************************************/
		
	//initialization of flow map as normal grid
	kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	
	//setting initial conditions for vorticity by translating with initial grid
	translate_initial_condition_through_map_stack(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
			cufftPlan_fine, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Temp_C1, Dev_Temp_C2);

	// compute first psi
	// compute stream hermite from vorticity, we have two different versions avaiable
	evaluate_stream_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, cufftPlan_vort, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());

    //evaulate_stream_hermite(Grid_2048, Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
    // set previous psi value to first psi value
	for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
		cudaMemcpyAsync(Dev_Psi_real + 4*Grid_psi.N*i_lagrange, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[i_lagrange]);
	}
	
	// save function to save variables, combined so we always save in the same way and location
    // use Dev_Hat_fine for W_fine, this works because just at the end of conservation it is overwritten
	if (SettingsMain.getSaveInitial()) {
		apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
				(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

		writeTimeStep(SettingsMain, "0", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, Grid_fine, Grid_coarse, Grid_psi);
		// compute conservation for first step
	}
	// compute conservation if wanted, not connected to normal saving to reduce data
	if (SettingsMain.getConvInitFinal()) {
		compute_conservation_targets(Grid_fine, Grid_coarse, Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);
		count_mesure++;
	}
	// sample if wanted
	if (SettingsMain.getSampleOnGrid() && SettingsMain.getSampleSaveInitial()) {
		sample_compute_and_write(Map_Stack, Grid_sample, Host_save, Dev_Temp_3,
				cufftPlan_sample, Dev_Temp_C1, Dev_Temp_C2,
				Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain, "0",
				Mesure_sample, count_mesure_sample);
		count_mesure_sample++;
	}

    cudaDeviceSynchronize();

    // now lets get the particles in
    if (SettingsMain.getParticles()) {

		// velocity initialization for inertial particles
		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p++){
			Particle_advect_inertia_init<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos + 2*Nb_particles*index_tau_p, Dev_particles_vel + 2*Nb_particles*index_tau_p, Dev_Psi_real, Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h);
		}

		if (SettingsMain.getParticlesSaveInitial()) {
			writeParticles(SettingsMain, "0", Host_particles_pos, Dev_particles_pos);
		}
	}

	// zoom if wanted, has to be done after particle initialization, maybe a bit useless at first instance
	if (SettingsMain.getZoom() && SettingsMain.getZoomSaveInitial()) {
		Zoom(SettingsMain, Map_Stack, Grid_zoom, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial,
				cufftPlan_fine, cufftPlan_zoom, Dev_Temp_3, Dev_Temp_C1, Dev_Temp_C2,
				Host_particles_pos, Dev_particles_pos, Host_save, "0");
	}


	// displaying max and min of vorticity and velocity for plotting limits and cfl condition
	// vorticity minimum
	if (SettingsMain.getVerbose() >= 2) {
		thrust::device_ptr<double> w_ptr = thrust::device_pointer_cast(Dev_W_H_fine_real);
		double w_max = thrust::reduce(w_ptr, w_ptr + Grid_fine.N, 0.0, thrust::maximum<double>());
		double w_min = thrust::reduce(w_ptr, w_ptr + Grid_fine.N, 0.0, thrust::minimum<double>());

		// velocity minimum - we first have to compute the norm elements
		thrust::device_ptr<double> psi_ptr = thrust::device_pointer_cast(Dev_W_H_fine_real);
		thrust::device_ptr<double> temp_ptr = thrust::device_pointer_cast((cufftDoubleReal*)Dev_Temp_C1);

		thrust::transform(psi_ptr + 1*Grid_psi.N, psi_ptr + 2*Grid_psi.N, psi_ptr + 2*Grid_psi.N, temp_ptr, norm_fun());
		double u_max = thrust::reduce(temp_ptr, temp_ptr + Grid_psi.N, 0.0, thrust::maximum<double>());

		message = "W min = " + to_str(w_min) + " - W max = " + to_str(w_max) + " - U max = " + to_str(u_max);
		cout<<message+"\n"; logger.push(message);
	}
	
	double t_vec[iterMax+SettingsMain.getLagrangeOrder()], dt_vec[iterMax+SettingsMain.getLagrangeOrder()];
	for (int i_l = 0; i_l < SettingsMain.getLagrangeOrder(); ++i_l) {
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
	if (SettingsMain.getVerbose() >= 2) {
		message = "Initialization Time = " + format_duration(time_values[loop_ctr]); cout<<message+"\n"; logger.push(message);
	}

	/*******************************************************************
	*						  Last Cuda Error						   *
	*******************************************************************/

	if (SettingsMain.getVerbose() >= 1) {
		message = cudaGetErrorName(cudaGetLastError()); cout<<message+"\n\n"; logger.push(message);
	}

	/*******************************************************************
	*							 Main loop							   *
	*******************************************************************/

	while(tf - t_vec[loop_ctr + SettingsMain.getLagrangeOrder()-1] > 1e-10 && loop_ctr < iterMax)
	{
		/*
		 * Timestep initialization:
		 *  - avoid overstepping (works with lagrange interpolation)
		 *  - save dt and t into vectors for lagrange interpolation
		 */
		//avoiding over-stepping for last time-step, this works fine with lagrange interpolation
		int loop_ctr_l = loop_ctr + SettingsMain.getLagrangeOrder()-1;
		if(t_vec[loop_ctr_l] + dt > tf) dt = tf - t_vec[loop_ctr_l];

		// set new dt into time vectors for lagrange interpolation
		t_vec[loop_ctr_l + 1] = t_vec[loop_ctr_l] + dt;
		dt_vec[loop_ctr_l + 1] = dt;

		/*
		 * Map advection
		 *  - Velocity is already intialized, so we can safely do that here
		 */
	    // by grid itself - only useful for very large grid sizes (has to be investigated)
	    if (SettingsMain.getMapUpdateGrid()) {
	    advect_using_stream_hermite_grid(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
	    		(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, Dev_Psi_real, t_vec, dt_vec, loop_ctr);
	    }
	    // by footpoints
	    else {
		    advect_using_stream_hermite(SettingsMain, Grid_coarse, Grid_psi, Dev_ChiX, Dev_ChiY,
		    		(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, Dev_Psi_real, t_vec, dt_vec, loop_ctr);
			cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)Dev_Temp_C1, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
			cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C2, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);
	    }


		/*******************************************************************
		*							 Remapping							   *
		*				Computing incompressibility
		*				Checking against threshold
		*				If exceeded: Safe map on CPU RAM
		*							 Initialize variables
		*******************************************************************/
		incomp_error[loop_ctr] = incompressibility_check(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Grid_fine, Grid_coarse);

		//resetting map and adding to stack
		if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) {

			//		if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) {
			if(Map_Stack.map_stack_ctr > Map_Stack.cpu_map_num*Map_Stack.Nb_array_RAM)
			{
				if (SettingsMain.getVerbose() >= 0) {
					message = "Stack Saturated : Exiting"; cout<<message+"\n"; logger.push(message);
				}
				break;
			}
			
			if (SettingsMain.getVerbose() >= 2) {
				message = "Refining Map : Step = " + to_str(loop_ctr) + " \t Maps = " + to_str(Map_Stack.map_stack_ctr) + " ; "
						+ to_str(Map_Stack.map_stack_ctr/Map_Stack.cpu_map_num) + " \t Gap = " + to_str(loop_ctr - old_ctr);
				cout<<message+"\n"; logger.push(message);
			}
			old_ctr = loop_ctr;
			
			//adjusting initial conditions, compute vorticity hermite
			translate_initial_condition_through_map_stack(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
					cufftPlan_fine, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Temp_C1, Dev_Temp_C2);
			
			if ((Map_Stack.map_stack_ctr%Map_Stack.cpu_map_num) == 0 && SettingsMain.getVerbose() >= 2){
				message = "Starting to use map stack array number " + to_str(Map_Stack.map_stack_ctr/Map_Stack.cpu_map_num);
				cout<<message+"\n"; logger.push(message);
			}

			Map_Stack.copy_map_to_host(Dev_ChiX, Dev_ChiY);
			
			//resetting map
			kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
		}
		// save map invetigative details
		map_gap[loop_ctr] = loop_ctr - old_ctr;
		map_ctr[loop_ctr] = Map_Stack.map_stack_ctr;
		
		//loop counters are increased after computation, only output is interested in that anyways
		loop_ctr ++;

	    /*
	     * Evaluation of stream hermite for the velocity at end of step to resemble velocity after timestep
	     *  - first copy the old values
	     *  - afterwards compute the new
	     *  - done before particles so that particles can benefit from nice timesteppings
	     */
        //copy Psi to Psi_previous and Psi_previous to Psi_previous_previous
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpy(Dev_Psi_real + 4*Grid_psi.N*(SettingsMain.getLagrangeOrder()-i_lagrange),
					   Dev_Psi_real + 4*Grid_psi.N*(SettingsMain.getLagrangeOrder()-i_lagrange-1), 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
		}

		// compute stream hermite from vorticity for next time step so that particles can benefit from it
		evaluate_stream_hermite(Grid_coarse, Grid_fine, Grid_psi, Grid_vort, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, cufftPlan_vort, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());


		/*
		 * Particles advection after velocity update to profit from nice avaiable accelerated schemes
		 */
		// Particles advection after velocity update to profit from nice avaiable schemes
	    if (SettingsMain.getParticles()) {
			Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real,
					Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, SettingsMain.getParticlesTimeIntegrationNum());
			// loop for all tau p
			for(int i_tau_p = 1; i_tau_p < SettingsMain.getParticlesTauNum(); i_tau_p+=1){
				Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt,
						Dev_particles_pos + 2*Nb_particles*i_tau_p, Dev_particles_vel + 2*Nb_particles*i_tau_p, Dev_Psi_real,
						Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, SettingsMain.particles_tau[i_tau_p], SettingsMain.getParticlesTimeIntegrationNum());
			}
			// copy fine particle positions
			if (SettingsMain.getParticlesSnapshotsPerSec() > 0 && SettingsMain.getSaveFineParticles()) {
				cudaMemcpy(Dev_particles_fine_pos + (loop_ctr*2*SettingsMain.getParticlesTauNum()*SettingsMain.getParticlesFineNum()*2) % (2*prod_fine_dt_particles), Dev_particles_pos, 2*SettingsMain.getParticlesTauNum()*SettingsMain.getParticlesFineNum()*sizeof(double), cudaMemcpyDeviceToDevice);
			}
	    }


			/*******************************************************************
			*							 Save snap shot						   *
			*				Normal variables in their respective grid
			*				Variables on sampled grid if wanted
			*				Particles together with fine particles
			*******************************************************************/
		if( loop_ctr % save_buffer_count == 0 )
		{
			if (SettingsMain.getVerbose() >= 2) {
				message = "Saving data : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
				cout<<message+"\n"; logger.push(message);
			}

			// save function to save variables, combined so we always save in the same way and location
			apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
					(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

			writeTimeStep(SettingsMain, to_str(t_vec[loop_ctr_l+1]), Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, Grid_fine, Grid_coarse, Grid_psi);
			// compute conservation for first step
			compute_conservation_targets(Grid_fine, Grid_coarse, Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);
		    count_mesure++;
		}
		if (SettingsMain.getSampleOnGrid() && loop_ctr % sample_save_buffer_count == 0) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Saving sample data : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
				cout<<message+"\n"; logger.push(message);
			}
	    	sample_compute_and_write(Map_Stack, Grid_sample, Host_save, Dev_Temp_3,
	    			cufftPlan_sample, Dev_Temp_C1, Dev_Temp_C2,
	    			Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain, to_str(t_vec[loop_ctr_l+1]),
	    			Mesure_sample, count_mesure_sample);
	    	count_mesure_sample++;
		}


		if (SettingsMain.getParticles() && loop_ctr % particles_save_buffer_count == 0) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Saving particles : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
				cout<<message+"\n"; logger.push(message);
			}

			// save particle positions
			if (SettingsMain.getParticles()) {
				writeParticles(SettingsMain, to_str(t_vec[loop_ctr_l+1]), Host_particles_pos, Dev_particles_pos);

				// if wanted, save fine particles, 1 file for all positions
				if (SettingsMain.getSaveFineParticles()) {
					writeFineParticles(SettingsMain, to_str(t_vec[loop_ctr_l+1]), Host_particles_fine_pos, Dev_particles_fine_pos, prod_fine_dt_particles);
				}
			}
		}
		
		// zoom if wanted, has to be done after particle initialization, maybe a bit useless at first instance
		if (SettingsMain.getZoom() && loop_ctr % zoom_save_buffer_count == 0) {
			if (SettingsMain.getVerbose() >= 2) {
				message = "Saving Zoom : T = " + to_str(t_vec[loop_ctr_l+1]) + " \t Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
				cout<<message+"\n"; logger.push(message);
			}

			Zoom(SettingsMain, Map_Stack, Grid_zoom, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial,
					cufftPlan_fine, cufftPlan_zoom, Dev_Temp_3, Dev_Temp_C1, Dev_Temp_C2,
					Host_particles_pos, Dev_particles_pos, Host_save, "0");
		}

		/*
		 * Some small things at the end of the loop
		 *  - check for errors
		 *  - spit out some verbose information
		 *  - safe timing of loop
		 */

		int error = cudaGetLastError();
		if(error != 0)
		{
			if (SettingsMain.getVerbose() >= 0) {
				message = "Exited early : Last Cuda Error = " + to_str(error); cout<<message+"\n"; logger.push(message);
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
			message = "Step = " + to_str(loop_ctr) + "/" + to_str(iterMax)
					+ " \t S-Time = " + to_str(t_vec[loop_ctr_l+1]) + "/" + to_str(tf)
					+ " \t IncompErr = " + to_str(incomp_error[loop_ctr-1])
					+ " \t C-Time = " + format_duration(time_values[loop_ctr]);
			cout<<message+"\n"; logger.push(message);
		}
	}
	
	
	// hackery: new loop for particles to test convergence without map influence
	if (SettingsMain.getParticlesSteps() != -1) {

		// copy velocity to old values to be uniform and constant in time
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpyAsync(Dev_Psi_real + 4*Grid_psi.N*i_lagrange, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[i_lagrange]);
		}
		// initialize particles again
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*SettingsMain.getParticlesTauNum());

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p+=1)
			cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, Dev_particles_pos);  // project 0-1 onto 0-LX

		for(int index_tau_p = 1; index_tau_p < SettingsMain.getParticlesTauNum(); index_tau_p+=1){
			Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
			Particle_advect_inertia_init<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h);
		}
		// save inital position to check if they are qual
		writeParticles(SettingsMain, "C0", Host_particles_pos, Dev_particles_pos);

		auto begin_P = std::chrono::high_resolution_clock::now();
		if (SettingsMain.getVerbose() >= 2) {
			message = "Starting extra particle loop : Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(begin_P - begin).count()/1e6);
			cout<<message+"\n"; logger.push(message);
		}

		// main loop until time 1
		double dt_p = 1/(double)SettingsMain.getParticlesSteps();
		for (int loop_ctr_p = 0; loop_ctr_p < SettingsMain.getParticlesSteps(); ++loop_ctr_p) {
			// particles advection
			Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt_p, Dev_particles_pos, Dev_Psi_real,
					Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, SettingsMain.getParticlesTimeIntegrationNum());
			// loop for all tau p
			for(int i_tau_p = 1; i_tau_p < SettingsMain.getParticlesTauNum(); i_tau_p+=1){
				Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt_p,
						Dev_particles_pos + 2*Nb_particles*i_tau_p, Dev_particles_vel + 2*Nb_particles*i_tau_p, Dev_Psi_real,
						Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, SettingsMain.particles_tau[i_tau_p], SettingsMain.getParticlesTimeIntegrationNum());
			}
		}

		// force synchronize after loop to wait until everything is finished
		cudaDeviceSynchronize();

		auto end_P = std::chrono::high_resolution_clock::now();
		if (SettingsMain.getVerbose() >= 2) {
			message = "Finished extra particle loop : Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(end_P - begin).count()/1e6);
			cout<<message+"\n"; logger.push(message);
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
	if (SettingsMain.getSaveFinal()) {
		apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
				(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

		writeTimeStep(SettingsMain, "final", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, Grid_fine, Grid_coarse, Grid_psi);
	}
	// compute conservation if wanted, not connected to normal saving to reduce data
	if (SettingsMain.getConvInitFinal()) {
		compute_conservation_targets(Grid_fine, Grid_coarse, Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);
		count_mesure++;
	}
	// sample if wanted
	if (SettingsMain.getSampleOnGrid() && SettingsMain.getSampleSaveFinal()) {
		sample_compute_and_write(Map_Stack, Grid_sample, Host_save, Dev_Temp_3,
				cufftPlan_sample, Dev_Temp_C1, Dev_Temp_C2,
				Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain, "final",
				Mesure_sample, count_mesure_sample);
		count_mesure_sample++;
	}
	if (SettingsMain.getParticles() && SettingsMain.getParticlesSaveFinal()) {
		writeParticles(SettingsMain, "final", Host_particles_pos, Dev_particles_pos);
	}

	// zoom if wanted
	if (SettingsMain.getZoom() && SettingsMain.getZoomSaveFinal()) {
		Zoom(SettingsMain, Map_Stack, Grid_zoom, Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Dev_W_H_initial,
				cufftPlan_fine, cufftPlan_zoom, Dev_Temp_3, Dev_Temp_C1, Dev_Temp_C2,
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

    // save imcomp error
	writeAllRealToBinaryFile(iterMax, incomp_error, SettingsMain, "/Monitoring_data/Incompressibility_check");
	// save map monitorings
	writeAllRealToBinaryFile(iterMax, map_gap, SettingsMain, "/Monitoring_data/Map_gaps");
	writeAllRealToBinaryFile(iterMax, map_ctr, SettingsMain, "/Monitoring_data/Map_counter");

	
	// save map stack if wanted
	if (SettingsMain.getSaveMapStack()) {
		// last map should be included too, so we have a full stack until this time
		if (Map_Stack.map_stack_ctr < Map_Stack.cpu_map_num*Map_Stack.Nb_array_RAM) {
			Map_Stack.copy_map_to_host(Dev_ChiX, Dev_ChiY);
		}
		if (SettingsMain.getVerbose() >= 2) {
			message = "Saving MapStack : Maps = " + to_str(Map_Stack.map_stack_ctr) + " \t Filesize = " + to_str(Map_Stack.map_stack_ctr*map_size) + "mb"; cout<<message+"\n"; logger.push(message);
		}
		writeMapStack(SettingsMain, Map_Stack);
	}

	
	/*******************************************************************
	*					  Zoom on the last frame					   *
	*******************************************************************/
	
//	Zoom(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack,
//			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//			Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM,
//			Dev_W_fine, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, simulationName, LX);
	
	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/
	
	cudaFree(Dev_W_H_initial);
	
	// Trash variable
	cudaFree(Dev_Temp_C1);
	cudaFree(Dev_Temp_C2);
	delete [] Host_save;
	
	// Chi
	cudaFree(Dev_ChiX);
	cudaFree(Dev_ChiY);

	// Chistack
	Map_Stack.free_res();
	
	// Vorticity
	cudaFree(Dev_W_coarse);
	cudaFree(Dev_W_H_fine_real);
	
	#ifdef DISCRET
		cudaFree(Dev_W_H_Initial);
	#endif

	// Psi
	cudaFree(Dev_Psi_real);

	// CuFFT plans
	cufftDestroy(cufftPlan_coarse);
	cufftDestroy(cufftPlan_fine);
	cufftDestroy(cufftPlan_psi);
	if (SettingsMain.getSampleOnGrid()) {
		cufftDestroy(cufftPlan_sample);
	}
	cudaFree(fft_work_area);

	if (SettingsMain.getParticles()) {
	// Particles variables
	    delete [] Host_particles_pos;
	    delete [] Host_particles_vel;
	    delete [] Host_particles_fine_pos;
	    cudaFree(Dev_particles_pos);
        cudaFree(Dev_particles_vel);
        cudaFree(Dev_particles_fine_pos);
	}

    cudaFree(Mesure);
    cudaFree(Mesure_fine);
	if (SettingsMain.getSampleOnGrid()) {
		cudaFree(Mesure_sample);
	}

	// save timing at last part of a step to take everything into account
    {
		auto step = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
		time_values[loop_ctr+1] = diff;
    }
    // save timing to file
	writeAllRealToBinaryFile(iterMax+2, time_values, SettingsMain, "/Monitoring_data/Timing_Values");

	if (SettingsMain.getVerbose() >= 1) {
		message = "Finished - Last Cuda Error = " + to_str(cudaGetLastError()) + " , Time = " + format_duration(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
		cout<<message+"\n"; logger.push(message);
	}
}


/*******************************************************************
*							 Remapping							   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D Grid_fine, MapStack Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufftPlan_fine, double *bounds, double *W_initial, int simulation_num_c,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2)
{
	
	// Vorticity on coarse grid to vorticity on fine grid with a very long command, use Dev_Temp_C1 for W_fine
	apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
			(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, W_initial, simulation_num_c);

	//kernel_apply_map_stack_to_W<<<Grid_fineblocksPerGrid, Grid_finethreadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, W_real, stack_length, Grid_coarseNX, Grid_coarseNY, Grid_coarseh, Grid_fineNX, Grid_fineNY, Grid_fineh, W_initial);
	
	k_real_to_comp<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, Grid_fine.NX, Grid_fine.NY);

	cufftExecZ2Z(cufftPlan_fine, Dev_Temp_C2, Dev_Temp_C1, CUFFT_FORWARD);
	k_normalize<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Temp_C1, Grid_fine.NX, Grid_fine.NY);

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3)
//	k_fft_cut_off_scale<<<Grid_fineblocksPerGrid, Grid_finethreadsPerBlock>>>(Dev_Temp_C1, Grid_fineNX, (double)(Grid_fineNX)/3.0);
	
	// form hermite formulation
	fourier_hermite(Grid_fine, Dev_Temp_C1, W_H_real, Dev_Temp_C2, cufftPlan_fine);
}


/*******************************************************************
*						 Computation of Psi						   *
*******************************************************************/

// upsample psi by doing zero padding vorticity in fourier space from coarse grid to psi grid
void evaluate_stream_hermite(TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi, TCudaGrid2D Grid_vort, double *Dev_ChiX, double *Dev_ChiY,
		double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_psi, cufftHandle cufftPlan_vort,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, int molly_stencil, double freq_cut_psi)
{

	// apply map to w and sample using mollifier, do it on a special grid for vorticity and apply mollification if wanted
	kernel_apply_map_and_sample_from_hermite<<<Grid_vort.blocksPerGrid, Grid_vort.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C2, Dev_W_H_fine_real, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_vort.NX, Grid_vort.NY, Grid_vort.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, molly_stencil);

	// forward fft
	k_real_to_comp<<<Grid_vort.blocksPerGrid, Grid_vort.threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C2, Dev_Temp_C1, Grid_vort.NX, Grid_vort.NY);
	cufftExecZ2Z(cufftPlan_vort, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid_vort.blocksPerGrid, Grid_vort.threadsPerBlock>>>(Dev_Temp_C2, Grid_vort.NX, Grid_vort.NY);

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3) and compute smooth W
	// use Psi grid here for intermediate storage
//	k_fft_cut_off_scale<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_Temp_C2, Grid_coarse.NX, (double)(Grid_psi.NX)/3.0);

	// save vorticity on coarse grid
	k_fft_grid_remove<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse.NX, Grid_vort.NX);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, (cufftDoubleComplex*)Psi_real, CUFFT_INVERSE);
	comp_to_real((cufftDoubleComplex*)Psi_real, W_real, Grid_coarse.N);

	// transition to stream function grid with three cases : grid_vort < grid_psi, grid_vort > grid_psi (a bit dumb) and grid_vort == grid_psi
	// first case : we have to do zero padding in the middle and increase the gridsize
	if (Grid_vort.NX < Grid_psi.NX) {
		cudaMemset(Dev_Temp_C1, 0, Grid_psi.sizeNComplex);
		k_fft_grid_add<<<Grid_vort.blocksPerGrid, Grid_vort.threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_vort.NX, Grid_psi.NX);
	}
	// second case : we have to remove entries and shrinken the grid
	else if (Grid_vort.NX > Grid_psi.NX) {
		k_fft_grid_remove<<<Grid_psi.blocksPerGrid, Grid_psi.threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_psi.NX, Grid_vort.NX);
	}
	// third case : do nothing if they are equal, copy data to not confuse all entries
	else { cudaMemcpy(Dev_Temp_C1, Dev_Temp_C2, Grid_psi.sizeNComplex, cudaMemcpyDeviceToDevice); }

	// cut high frequencies in fourier space, however not that much happens after zero move add from coarse grid
	k_fft_cut_off_scale<<<Grid_psi.blocksPerGrid, Grid_psi.threadsPerBlock>>>(Dev_Temp_C1, Grid_psi.NX, freq_cut_psi);

	// Forming Psi hermite now on psi grid
	k_fft_iLap<<<Grid_psi.blocksPerGrid, Grid_psi.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid_psi.NX, Grid_psi.NY, Grid_psi.h);												// Inverse laplacian in Fourier space
	fourier_hermite(Grid_psi, Dev_Temp_C2, Psi_real, Dev_Temp_C1, cufftPlan_psi);
}
// debugging lines, could be needed here to check psi
//	cudaMemcpy(Host_Debug, Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(4*Grid_psi.N, Host_Debug, "psi_debug_4_nodes_C512_F2048_t64_T1", "Debug_2");


// sample psi on a fixed grid with vorticity known
void psi_upsampling(TCudaGrid2D Grid, double *Dev_W, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, double *Dev_Psi, cufftHandle cufftPlan){
	k_real_to_comp<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_W, Dev_Temp_C1, Grid.NX, Grid.NY);

	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, Grid.NX, Grid.NY);

	// Forming Psi hermite
	k_fft_iLap<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid.NX, Grid.NY, Grid.h);													// Inverse laplacian in Fourier space
	fourier_hermite(Grid, Dev_Temp_C1, Dev_Psi, Dev_Temp_C2, cufftPlan);
}


// compute hermite with derivatives in fourier space, uniform helper function fitted for all grids to utilize only two trash variables
// input is given in first trash variable
void fourier_hermite(TCudaGrid2D Grid, cufftDoubleComplex *Dev_Temp_C1, double *Dev_Output, cufftDoubleComplex *Dev_Temp_C2, cufftHandle cufftPlan) {
	// dy and dxdy derivates are stored in later parts of output array, we can therefore use the first half as a trash variable
	// start with dy derivative and store in position 3/4
	k_fft_dy<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid.NX, Grid.NY, Grid.h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan, Dev_Temp_C2, (cufftDoubleComplex*)Dev_Output, CUFFT_INVERSE);
	comp_to_real((cufftDoubleComplex*)Dev_Output, Dev_Output + 2*Grid.N, Grid.N);

	// reuse values from Trash_C2 and create dxdy, store in position 4/4
	k_fft_dx<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C2, (cufftDoubleComplex*)Dev_Output, Grid.NX, Grid.NY, Grid.h);
	cufftExecZ2Z(cufftPlan, (cufftDoubleComplex*)Dev_Output, Dev_Temp_C2, CUFFT_INVERSE);
	comp_to_real(Dev_Temp_C2, Dev_Output + 3*Grid.N, Grid.N);

	// we can go back to Trash_C1 and store the other two values normally, first normal values
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);
	comp_to_real(Dev_Temp_C2, Dev_Output, Grid.N);

	// now compute dx derivative and store it
	k_fft_dx<<<Grid.blocksPerGrid, Grid.threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid.NX, Grid.NY, Grid.h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan, Dev_Temp_C2, Dev_Temp_C1, CUFFT_INVERSE);
	comp_to_real(Dev_Temp_C1, Dev_Output + Grid.N, Grid.N);
}


/*******************************************************************
*		 Computation of Global conservation values				   *
*******************************************************************/

void compute_conservation_targets(TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_fine,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2,
		double *Mesure, double *Mesure_fine, int count_mesure)
{
	// coarse grid
	Compute_Energy(&Mesure[3*count_mesure], Dev_Psi, Grid_psi);
	Compute_Enstrophy(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse);
	// fine grid, no energy because we do not have velocity on fine grid
	Compute_Enstrophy(&Mesure_fine[1 + 3*count_mesure], Dev_W_H_fine, Grid_fine);

	// palinstrophy, fine first because vorticity is saved in temporal array
	// fine palinstrophy cannot be computed, as we have Dev_W_H_fine in Tev_Temp_C1, which is needed
//	Compute_Palinstrophy(Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_H_fine, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_fine);
	Compute_Palinstrophy(Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_coarse);
}


/*******************************************************************
*		 Sample on a specific grid and save everything	           *
*******************************************************************/

void sample_compute_and_write(MapStack Map_Stack, TCudaGrid2D Grid_sample, double *Host_sample, double *Dev_sample,
		cufftHandle cufftPlan_sample, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2,
		double *Dev_ChiX, double*Dev_ChiY, double *bounds, double *W_initial, SettingsCMM SettingsMain, string i_num,
		double *Mesure_sample, int count_mesure) {

	// begin with vorticity
	apply_map_stack_to_W_part_All(Grid_sample, Map_Stack, Dev_ChiX, Dev_ChiY,
			Dev_sample, (cufftDoubleReal*)Dev_Temp_C1, bounds, W_initial, SettingsMain.getInitialConditionNum());
	writeTimeVariable(SettingsMain, "Vorticity_W_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample, Grid_sample.sizeNReal, Grid_sample.N);

	// compute enstrophy and palinstrophy
	Compute_Enstrophy(&Mesure_sample[1 + 3*count_mesure], Dev_sample, Grid_sample);
	Compute_Palinstrophy(Grid_sample, &Mesure_sample[2 + 3*count_mesure], Dev_sample, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_sample);

	// reuse sampled vorticity to compute psi
	psi_upsampling(Grid_sample, Dev_sample, Dev_Temp_C1, Dev_Temp_C2, Dev_sample, cufftPlan_sample);
	writeTimeVariable(SettingsMain, "Stream_Function_Psi_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample, 4*Grid_sample.sizeNReal, 4*Grid_sample.N);

	// compute energy
	Compute_Energy(&Mesure_sample[3*count_mesure], Dev_sample, Grid_sample);

	// map
	k_sample<<<Grid_sample.blocksPerGrid,Grid_sample.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_sample, Dev_sample + Grid_sample.N, Map_Stack.Grid->NX, Map_Stack.Grid->NY, Map_Stack.Grid->h, Grid_sample.NX, Grid_sample.NY, Grid_sample.h);
	writeTimeVariable(SettingsMain, "Map_ChiX_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample, Grid_sample.sizeNReal, Grid_sample.N);
	writeTimeVariable(SettingsMain, "Map_ChiY_"+to_str(Grid_sample.NX), i_num, Host_sample, Dev_sample + Grid_sample.N, Grid_sample.sizeNReal, Grid_sample.N);
}



/*******************************************************************
*							   Zoom								   *
*******************************************************************/


void Zoom(SettingsCMM SettingsMain, MapStack Map_Stack, TCudaGrid2D Grid_zoom, double *Dev_ChiX, double *Dev_ChiY,
		double *W_real, double *W_initial, cufftHandle cufftPlan_fine, cufftHandle cufftPlan_zoom,
		double *Dev_Temp_3, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2,
		double *Host_particles_pos, double *Dev_particles_pos, double *Host_debug, string i_num)
{
	// create folder
	string sub_folder_name = "/Zoom_data/Time_" + i_num;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	mkdir(folder_name_now.c_str(), 0700);

	double x_min, x_max, y_min, y_max;

	double x_width = SettingsMain.getZoomWidthX()/2.0;
	double y_width = SettingsMain.getZoomWidthY()/2.0;

	// do repetetive zooms
	for(int zoom_ctr = 0; zoom_ctr < SettingsMain.getZoomRepetitions(); zoom_ctr++){
		// create new subfolder for current zoom
		sub_folder_name = "/Zoom_data/Time_" + i_num + "/Zoom_" + to_str(zoom_ctr);
		folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
		mkdir(folder_name_now.c_str(), 0700);

		// construct frame bounds for this zoom
		x_min = SettingsMain.getZoomCenterX() - x_width;
		x_max = SettingsMain.getZoomCenterX() + x_width;
		y_min = SettingsMain.getZoomCenterY() - y_width;
		y_max = SettingsMain.getZoomCenterY() + y_width;
		// safe bounds in array
		double bounds[4] = {x_min, x_max, y_min, y_max};

		// compute zoom of vorticity
		apply_map_stack_to_W_part_All(Grid_zoom, Map_Stack, Dev_ChiX, Dev_ChiY,
				W_real, Dev_Temp_3, bounds, W_initial, SettingsMain.getInitialConditionNum());

		// save vorticity zoom
		cudaMemcpy(Host_debug, W_real, Grid_zoom.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_zoom.N, Host_debug, SettingsMain, sub_folder_name+"/Vorticity_W");

		// compute zoom of stream function, this uses the zoomed vorticity
		if (SettingsMain.getZoomSavePsi()) {
			psi_upsampling(Grid_zoom, W_real, Dev_Temp_C1, Dev_Temp_C2, Dev_Temp_3, cufftPlan_zoom);
			// save psi zoom
			cudaMemcpy(Host_debug, Dev_Temp_3, Grid_zoom.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_zoom.N, Host_debug, SettingsMain, sub_folder_name+"/Stream_function_Psi");
		}


		// safe particles in zoomframe if wanted
		// copy particles to host
	    cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*SettingsMain.getParticlesNum()*SettingsMain.getParticlesTauNum()*sizeof(double), cudaMemcpyDeviceToHost);
	    // loop over all tau numbers
		if (SettingsMain.getParticles() && SettingsMain.getZoomSaveParticles()) {
			// primitive loop on host, maybe this could be implemented more clever, but ca marche
			for (int i_tau = 0; i_tau < SettingsMain.getParticlesTauNum(); i_tau++) {
				int part_counter = 0;
				int tau_shift = 2*i_tau*SettingsMain.getParticlesNum();
				for (int i_p = 0; i_p < SettingsMain.getParticlesNum(); i_p++) {
					// check if particle in frame and then save it inside itself
					if (Host_particles_pos[2*i_p   + tau_shift] > x_min &&
						Host_particles_pos[2*i_p   + tau_shift] < x_max &&
						Host_particles_pos[2*i_p+1 + tau_shift] > y_min &&
						Host_particles_pos[2*i_p+1 + tau_shift] < y_max) {
						// transcribe particle
						Host_particles_pos[2*part_counter   + tau_shift] = Host_particles_pos[2*i_p   + tau_shift];
						Host_particles_pos[2*part_counter+1 + tau_shift] = Host_particles_pos[2*i_p+1 + tau_shift];
						// increment counter
						part_counter++;
					}
				}
				// save particles
				string tau_name;
				if (i_tau == 0) tau_name = "Fluid"; else tau_name = "Tau=" + to_str(SettingsMain.particles_tau[i_tau]);
				writeAllRealToBinaryFile(2*part_counter, Host_particles_pos+tau_shift, SettingsMain, sub_folder_name+"/Particles_pos_"+ tau_name);
			}
		}

		x_width *=  SettingsMain.getZoomRepetitionsFactor();
		y_width *=  SettingsMain.getZoomRepetitionsFactor();
	}
}


// helper function to format time to readable format
string format_duration(double sec) {
	return to_str(floor(sec/3600.0)) + "h " + to_str(floor(std::fmod(sec, 3600)/60.0)) + "m " + to_str(std::fmod(sec, 60)) + "s";
}


/*******************************************************************
*				Zoom for a specific time instant				   *
*******************************************************************/

// We have to check that it still works.
/*
void Zoom_load_frame(string File, int grid_scale, int fine_grid_scale, string t_nb){
	
	
	double LX;
	int NXc, NYc;														
	int NXsf, NYsf;														
	int map_stack_ctr = 23;									// don't need it, it can be tertemined by the size of data loaded...
	
	LX = twoPI;	
	NXc = NYc = grid_scale;
	NXsf = NYsf = fine_grid_scale;
	
	string simulationName = File;
	
	TCudaGrid2D Gc(NXc, NYc, LX);
	TCudaGrid2D Gsf(NXsf, NYsf, LX);
	
	
	double *ChiX, *ChiY, *ChiX_stack, *ChiY_stack;
	ChiX = new double[4*grid_scale*grid_scale];
	ChiY = new double[4*grid_scale*grid_scale];
	ChiX_stack = new double[map_stack_ctr * 4*Grid_coarse.sizeNReal];
	ChiY_stack = new double[map_stack_ctr * 4*Grid_coarse.sizeNReal];
	
	
	readAllRealFromBinaryFile(4*grid_scale*grid_scale, ChiX, simulationName, "ChiX_" + t_nb);
	readAllRealFromBinaryFile(4*grid_scale*grid_scale, ChiY, simulationName, "ChiY_" + t_nb);
	readAllRealFromBinaryFile(map_stack_ctr * 4*grid_scale*grid_scale, ChiX_stack, simulationName, "ChiX_stack_" + t_nb);
	readAllRealFromBinaryFile(map_stack_ctr * 4*grid_scale*grid_scale, ChiY_stack, simulationName, "ChiY_stack_" + t_nb);
	
	
	double *Dev_W_fine;
	cudaMalloc((void**)&Dev_W_fine,  Grid_fine.sizeNReal);
	
	double *Dev_ChiX, *Dev_ChiY;
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	
	double *Dev_ChiX_stack, *Dev_ChiY_stack;
	cudaMalloc((void **) &Dev_ChiX_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal);
	cudaMalloc((void **) &Dev_ChiY_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal);
	
	
	cudaMemcpy(Dev_ChiX, ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiY, ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiX_stack, ChiX_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_ChiY_stack, ChiY_stack, map_stack_ctr * 4*Grid_coarse.sizeNReal, cudaMemcpyHostToDevice);
	
	
	Zoom(simulationName, LX, Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Dev_W_fine, map_stack_ctr);
	
	
	delete [] ChiX;
	delete [] ChiY;
	
	cudaFree(Dev_W_fine);
	cudaFree(Dev_ChiX);
	cudaFree(Dev_ChiY);
	cudaFree(Dev_ChiX_stack);
	cudaFree(Dev_ChiY_stack);
	
	
	printf("Finished\n");
	
}
*/













	/**************************************************************************************************************************************
	*						   Some comments						   
	* 
	* 	- We can Dev_ChiX_stack from the Host to Dev because it is not always used. We will have more GPU memory.
	* 	- We can remove complex variables for more memory. 
	* 	- Parallel computing for FFT and Hermite interpolation
	*	
	* 
	**************************************************************************************************************************************/



























