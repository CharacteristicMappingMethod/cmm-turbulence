#include "cudaeuler2d.h"


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
	int NX_psi = SettingsMain.getGridPsi();								// psi grid size
	int NY_psi = SettingsMain.getGridPsi();								// psi grid size
	
	double t0 = 0.0;													// time - initial
	double tf = SettingsMain.getFinalTime();							// time - final
	double grid_by_time = SettingsMain.getFactorDtByGrid();				// time - factor by grid
	int steps_per_sec = SettingsMain.getStepsPerSec();					// time - steps per second
	double dt;															// time - step final
	int iterMax;														// time - maximum iteration count

	string workspace = SettingsMain.getWorkspace();						// folder where we work in
	string sim_name = SettingsMain.getSimName();						// name of the simulation
	string initial_condition = SettingsMain.getInitialCondition();		// name of the initial condition
	string file_name;
	int snapshots_per_second = SettingsMain.getSnapshotsPerSec();		// saves per second
	int save_buffer_count;												// iterations after which files should be saved
	int map_stack_length;												// this parameter is set to avoide memory overflow on GPU
	int show_progress_at;
	
	//GPU dependent parameters
	int mem_RAM_GPU_remaps = SettingsMain.getMemRamGpuRemaps(); 		// mem_index in MB on the GPU
	int mem_RAM_CPU_remaps = SettingsMain.getMemRamCpuRemaps();			// mem_RAM_CPU_remaps in MB on the CPU
	int Nb_array_RAM = 4;												// fixed for four different stacks
	int use_set_grid = 0;  												// change later, use 2048 grid thingys or not

//	double mb_used_RAM_CPU;
	double mb_used_RAM_GPU;  // count memory usage by variables in mbyte

	// compute dt, for grid settings we have to use max in case we want to differ NX and NY
	if (SettingsMain.getSetDtBySteps()) dt = 1.0 / steps_per_sec;  // direct setting of timestep
	else dt = 1.0 / std::max(NX_coarse, NY_coarse) / grid_by_time;  // setting of timestep by grid and factor
	
	double *Dev_W_H_initial; cudaMalloc((void**)Dev_W_H_initial, 8);

	#ifdef DISCRET
		grid_by_time = 8.0;
		snapshots_per_second = 1;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 100;	
	#endif
	
	//shared parameters
	iterMax = ceil(tf / dt);
	if (snapshots_per_second > 0) {
		save_buffer_count = 1.0/snapshots_per_second/dt;  // normal case
	}
	else {
		save_buffer_count = INT_MIN;  // disable saving
	}
	show_progress_at = (32 * 4 * pow(128, 3.0)) / pow(NX_coarse, 3.0);
		if(show_progress_at < 1) show_progress_at = 1;
	map_stack_length = (mem_RAM_GPU_remaps * pow(128, 2.0))/ (double(NX_coarse * NX_coarse));
	int frac_mem_cpu_to_gpu = int(double(mem_RAM_CPU_remaps)/double(mem_RAM_GPU_remaps)/double(Nb_array_RAM));  // define how many more remappings we can save on CPU than on GPU
	
	file_name = sim_name + "_" + initial_condition + "_C" + to_str(NX_coarse) + "_F" + to_str(NX_fine) + "_t" + to_str(1.0/dt).substr(0, std::to_string(1.0/dt).find(".")) + "_T" + to_str(tf).substr(0, to_str(tf).find("."));
	create_directory_structure(SettingsMain, file_name, dt, save_buffer_count, show_progress_at, iterMax, map_stack_length);
    Logger logger(file_name);

    string message;  // string to be used for console output
	message = "Initial condition = " + initial_condition; cout<<message+"\n"; logger.push(message);
	message = "Iter max = " + to_str(iterMax); cout<<message+"\n"; logger.push(message);
	message = "Save buffer count = " + to_str(save_buffer_count); cout<<message+"\n"; logger.push(message);
	message = "Progress at = " + to_str(show_progress_at); cout<<message+"\n"; logger.push(message);
	message = "Map stack length = " + to_str(map_stack_length); cout<<message+"\n"; logger.push(message);
	message = "Map stack length on RAM = " + to_str(frac_mem_cpu_to_gpu * map_stack_length); cout<<message+"\n"; logger.push(message);
	message = "Map stack length total on RAM = " + to_str(frac_mem_cpu_to_gpu * map_stack_length * Nb_array_RAM); cout<<message+"\n"; logger.push(message);
    message = "Name of simulation = " + file_name; cout<<message+"\n"; logger.push(message);
	
	
	/*******************************************************************
	*							Grids								   *
	* 	Coarse grid where we compute derivatives				       *
	* 	Fine grid where we interpolate using Hermite Basis functions   *
	* 	Psi grid on which we compute the stream function and velocity  *
	* 	Sample grid on which we want to save and sample results        *
	*																   *
	*******************************************************************/
	
	TCudaGrid2D Grid_coarse(NX_coarse, NY_coarse, LX);
	TCudaGrid2D Grid_fine(NX_fine, NY_fine, LX);
	TCudaGrid2D Grid_psi(NX_psi, NY_psi, LX);
	TCudaGrid2D Grid_sample(SettingsMain.getGridSample(), SettingsMain.getGridSample(), LX);
	
	
	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 	we never run ffts in parallel, so we can reuse the temp space  *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufftPlan_coarse, cufftPlan_fine, cufftPlan_psi, cufftPlan_sample;
	// preinitialize handles
	cufftCreate(&cufftPlan_coarse); cufftCreate(&cufftPlan_fine);
	cufftCreate(&cufftPlan_psi); cufftCreate(&cufftPlan_sample);

	// disable auto workspace creation for fft plans
	cufftSetAutoAllocation(cufftPlan_coarse, 0); cufftSetAutoAllocation(cufftPlan_fine, 0);
	cufftSetAutoAllocation(cufftPlan_psi, 0); cufftSetAutoAllocation(cufftPlan_sample, 0);

	// create plans and compute needed size of each plan
	size_t workSize[4];
	cufftMakePlan2d(cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z, &workSize[0]);
	cufftMakePlan2d(cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z, &workSize[1]);
	cufftMakePlan2d(cufftPlan_psi, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z, &workSize[2]);
	if (SettingsMain.getSampleOnGrid()) {
		cufftMakePlan2d(cufftPlan_sample, Grid_sample.NX, Grid_sample.NY, CUFFT_Z2Z, &workSize[3]);
	}

    // allocate memory to new workarea for cufft plans with maximum size
	size_t size_max_fft = std::max(workSize[0], workSize[1]);
	size_max_fft = std::max(size_max_fft, workSize[2]);
	if (SettingsMain.getSampleOnGrid()) {
		size_max_fft = std::max(size_max_fft, workSize[3]);
	}
	void *fft_work_area;
	cudaMalloc(&fft_work_area, size_max_fft);
	mb_used_RAM_GPU = size_max_fft/(double)(1024*1024);

	// set new workarea to plans
	cufftSetWorkArea(cufftPlan_coarse, fft_work_area); cufftSetWorkArea(cufftPlan_fine, fft_work_area);
	cufftSetWorkArea(cufftPlan_psi, fft_work_area);
	if (SettingsMain.getSampleOnGrid()) {
		cufftSetWorkArea(cufftPlan_sample, fft_work_area);
	}

////	size_t workSize;
//	cufftPlan2d(&cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z);
////	cufftGetSize(cufftPlan_coarse, &workSize);
////	mb_used_RAM_GPU += workSize/(double)(1024*1024);
//	cufftPlan2d(&cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z);
////	cufftGetSize(cufftPlan_fine, &workSize);
////	mb_used_RAM_GPU += workSize/(double)(1024*1024);
//	cufftPlan2d(&cufftPlan_psi, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z);
////	cufftGetSize(cufftPlan_psi, &workSize);
////	mb_used_RAM_GPU += workSize/(double)(1024*1024);
//	if (SettingsMain.getSampleOnGrid()) {
//		cufftPlan2d(&cufftPlan_sample, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z);
////		cufftGetSize(cufftPlan_sample, &workSize);
////		mb_used_RAM_GPU += workSize/(double)(1024*1024);
//	}
	
	
	/*******************************************************************
	*							Trash variable
	*          Used for temporary copying and storing of data
	*                       in various locations
	*  casting (cufftDoubleReal*) makes them usable for double arrays
	*******************************************************************/
	
	// set after largest grid, checks are for failsave in case one chooses weird grid
	long int size_max_c = std::max(Grid_fine.sizeNComplex, Grid_psi.sizeNComplex);
	size_max_c = std::max(size_max_c, 4*Grid_coarse.sizeNComplex);
	if (SettingsMain.getSampleOnGrid()) {
		size_max_c = std::max(size_max_c, Grid_sample.sizeNComplex);
	}
	// for now three thrash variables are needed for cufft with hermites
	cufftDoubleComplex *Dev_Temp_C1, *Dev_Temp_C2;
	cudaMalloc((void**)&Dev_Temp_C1, size_max_c);
	cudaMalloc((void**)&Dev_Temp_C2, size_max_c);
	mb_used_RAM_GPU += 2*size_max_c/(double)(1024*1024);
	
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
	mb_used_RAM_GPU += 8*Grid_coarse.sizeNReal/(double)(1024*1024);
	
	
	/*******************************************************************
	*					       Chi_stack							   *
	* 	We need to save the variable Chi to be able to make	the        *
	* 	remapping or the zoom				   					       *
	* 																   *
	*******************************************************************/
	
	MapStack Map_Stack(&Grid_coarse, frac_mem_cpu_to_gpu, map_stack_length);
	mb_used_RAM_GPU += 8*Grid_coarse.sizeNReal/(double)(1024*1024);
	message = "Map Stack Initialized"; cout<<message+"\n"; logger.push(message);
	
	
	/*******************************************************************
	*					       Vorticity							   *
	* 	We need to have different variable version. coarse/fine,       *
	* 	real/complex/hat and an array that contains NE, SE, SW, NW	   *
	* 																   *
	*******************************************************************/
	
	double *Dev_W_coarse, *Dev_W_H_fine_real;
	cudaMalloc((void**)&Dev_W_coarse, Grid_coarse.sizeNReal);
	mb_used_RAM_GPU += Grid_coarse.sizeNReal/(double)(1024*1024);
	//vorticity hermite
	cudaMalloc((void**)&Dev_W_H_fine_real, 4*Grid_fine.sizeNReal);
	mb_used_RAM_GPU += 4*Grid_fine.sizeNReal/(double)(1024*1024);
	
	
	/*******************************************************************
	*							DISCRET								   *
	*******************************************************************/
	
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
	mb_used_RAM_GPU += 4*SettingsMain.getLagrangeOrder()*Grid_psi.sizeNReal/(double)(1024*1024);

	
	/*******************************************************************
	*						Gradient of Chi							   *
	* 	We use the gradient of Chi to be sure that the flow is 	       *
	* 	still incompressible 										   *
	* 																   *
	*******************************************************************/
	
	double w_min, w_max;
	
	double grad_chi_min, grad_chi_max;
	
	int grad_block = 32, grad_thread = 1024; // settings for min/max function, maximum threads and just one block
	double *Host_w_min, *Host_w_max;
	double *Dev_w_min, *Dev_w_max;
	Host_w_min = new double[grad_block*grad_thread];
	Host_w_max = new double[grad_block*grad_thread];
	cudaMalloc((void**) &Dev_w_min, sizeof(double)*grad_block*grad_thread);
	cudaMalloc((void**) &Dev_w_max, sizeof(double)*grad_block*grad_thread);
	mb_used_RAM_GPU += 2*sizeof(double)*grad_block*grad_thread/(double)(1024*1024);
	
	
	/*******************************************************************
	*							 Particles							   *
	*******************************************************************/
	// all variables have to be defined outside
	int Nb_particles, particle_thread, particle_block;
	double *Host_particles_pos ,*Dev_particles_pos;  // position of particles
	double *Host_particles_vel ,*Dev_particles_vel;  // velocity of particles
	curandGenerator_t prng;
	int freq_fine_dt_particles, prod_fine_dt_particles;
	double *Dev_particles_fine_pos, *Host_particles_fine_pos;
	// i still dont really get pointers and arrays in c++, so this array will have to stay here for now
	int Nb_Tau_p = 2;
//	double Tau_p[Nb_Tau_p] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.25, 0.5, 0.75, 1, 2, 5, 13};
	double Tau_p[Nb_Tau_p] = {0.0, 1};
	// now set the variables if we have particles
	if (SettingsMain.getParticles()) {
		// initialize all memory
		Nb_particles = SettingsMain.getParticlesNum();
		particle_thread =  256;  // threads for particles, seems good like that
		particle_block = Nb_particles / particle_thread + (Nb_particles < particle_thread);  // we need atleast 1 block
		Host_particles_pos = new double[2*Nb_particles*Nb_Tau_p];
		Host_particles_vel = new double[2*Nb_particles*Nb_Tau_p];
		cudaMalloc((void**) &Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(double));
		cudaMalloc((void**) &Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(double));
		mb_used_RAM_GPU += 4*Nb_particles*Nb_Tau_p*sizeof(double)/(double)(1024*1024);

		// create initial positions from random distribution
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p);

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
			cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		// particles where every time step the position will be saved
		if (snapshots_per_second > 0 && SettingsMain.getSaveFineParticles()) {
			SettingsMain.setParticlesFineNum(std::min(SettingsMain.getParticlesFineNum(), SettingsMain.getParticlesNum()));
			freq_fine_dt_particles = save_buffer_count; // save particles at every saving step
			prod_fine_dt_particles = SettingsMain.getParticlesFineNum() * freq_fine_dt_particles;

			Host_particles_fine_pos = new double[2*prod_fine_dt_particles];
			cudaMalloc((void**) &Dev_particles_fine_pos, 2*prod_fine_dt_particles*Nb_Tau_p*sizeof(double));
			mb_used_RAM_GPU += 2*prod_fine_dt_particles*Nb_Tau_p*sizeof(double)/(double)(1024*1024);
		}

		create_particle_directory_structure(SettingsMain, file_name, Tau_p, Nb_Tau_p);

		message = "Number of particles = " + to_str(Nb_particles); cout<<message+"\n"; logger.push(message);
		if (snapshots_per_second > 0 && SettingsMain.getSaveFineParticles()) {
			message = "Number of fine particles = " + to_str(SettingsMain.getParticlesFineNum()); cout<<message+"\n"; logger.push(message);
		}
	}


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	int count_mesure = 0;
	int mes_size;
	if (snapshots_per_second > 0) {
		mes_size = (int)(tf*snapshots_per_second) + 2;  // add initial and last step
	}
	else {
		mes_size = 2;  // add initial and last step
	}
    double *Mesure, *Mesure_fine, *Mesure_sample;
	cudaMallocManaged(&Mesure, 3*mes_size*sizeof(double));
	cudaMallocManaged(&Mesure_fine, 3*mes_size*sizeof(double));

	if (SettingsMain.getSampleOnGrid()) {
		cudaMallocManaged(&Mesure_sample, 3*mes_size*sizeof(double));
	}

    double incomp_error [iterMax];  // save incompressibility error for investigations
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
	
	double *Dev_save_sample;

	if (SettingsMain.getSampleOnGrid()) {
		cudaMalloc((void**)&Dev_save_sample, 4*Grid_sample.sizeNReal);
		mb_used_RAM_GPU += 4*Grid_sample.sizeNReal/(double)(1024*1024);
    }





	/*******************************************************************
	*						   Initialization						   *
	*******************************************************************/
		
	//initialization of flow map as normal grid
	kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	
	//setting initial conditions for vorticity by translating with initial grid
	translate_initial_condition_through_map_stack(&Grid_fine, &Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
			cufftPlan_fine, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Temp_C1, Dev_Temp_C2);

	// compute first psi
	// compute stream hermite from vorticity, we have two different versions avaiable
	if (SettingsMain.getUpsampleVersion() == 0) {
		evaluate_stream_hermite(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());
	}
	else {
		evaluate_stream_hermite_2(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi(), Host_save);
	}
    //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
    // set previous psi value to first psi value
	for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
		cudaMemcpyAsync(Dev_Psi_real + 4*Grid_psi.N*i_lagrange, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[i_lagrange]);
	}
	
	// save function to save variables, combined so we always save in the same way and location
    // use Dev_Hat_fine for W_fine, this works because just at the end of conservation it is overwritten
	apply_map_stack_to_W_part_All(&Grid_fine, &Map_Stack, Dev_ChiX, Dev_ChiY,
			(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

	writeTimeStep(workspace, file_name, "0", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
    // compute conservation for first step
    compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);

    // sample if wanted
    if (SettingsMain.getSampleOnGrid()) {
    	sample_compute_and_write(&Map_Stack, &Grid_sample, Host_save, Dev_save_sample,
    			cufftPlan_sample, Dev_Temp_C1, Dev_Temp_C2,
    			Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(),
    			workspace, file_name, "0",
    			Mesure_sample, count_mesure);
    }


    count_mesure+=1;

    cudaDeviceSynchronize();

    // now lets get the particles in
    if (SettingsMain.getParticles()) {

		// Particles initialization
		Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, Dev_particles_pos);  // project 0-1 onto 0-LX

		for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
			Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
			Particle_advect_inertia_init<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h);
		}

    	writeParticles(SettingsMain, file_name, "0", Host_particles_pos, Dev_particles_pos, Tau_p, Nb_Tau_p);
	}




	/////////////////////// slight different from regular loop
	//saving max and min for plotting purpose
//    cudaMemcpy(Host_save, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
    cudaMemcpy(Host_save, Dev_W_H_fine_real, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	get_max_min(&Grid_fine, Host_save, &w_min, &w_max);
	message = "W min = " + to_str(w_min) + " - W max = " + to_str(w_max); cout<<message+"\n"; logger.push(message);
	
	double t = t0;
	int loop_ctr = 0;
	int save_ctr = 1;

	int old_ctr = 0;

	// print estimated gpu memory usage in mb
	message = "estimated GPU RAM usage in mb = " + to_str(mb_used_RAM_GPU); cout<<message+"\n"; logger.push(message);


	// first timing save before loop - this is the initialization time
	{
		auto step = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
		time_values[loop_ctr] = diff; // loop_ctr was already increased
	}
	message = "Initialization Time = " + to_str(time_values[loop_ctr]); cout<<message+"\n"; logger.push(message);

	/*******************************************************************
	*						  Last Cuda Error						   *
	*******************************************************************/

	message = cudaGetErrorName(cudaGetLastError()); cout<<message+"\n\n"; logger.push(message);

	/*******************************************************************
	*							 Main loop							   *
	*******************************************************************/

	while(tf - t > 1e-10 && loop_ctr < iterMax)
	{
		//avoiding over-stepping for last time-step
		if(t + dt > tf)
			dt = tf - t;

		// compute stream hermite from vorticity, we have two different versions avaiable
		if (SettingsMain.getUpsampleVersion() == 0) {
			evaluate_stream_hermite(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());
		}
		else {
			evaluate_stream_hermite_2(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi(), Host_save);
		}
        //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);


		// Particles advection
	    if (SettingsMain.getParticles()) {
			Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real,
					Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, SettingsMain.getParticlesTimeIntegrationNum());
			// loop for all tau p
			for(int i_tau_p = 1; i_tau_p < Nb_Tau_p; i_tau_p+=1){
				Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt,
						Dev_particles_pos + 2*Nb_particles*i_tau_p, Dev_particles_vel + 2*Nb_particles*i_tau_p, Dev_Psi_real,
						Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, Tau_p[i_tau_p], SettingsMain.getParticlesTimeIntegrationNum());
			}
			// copy fine particle positions
			if (snapshots_per_second > 0 && SettingsMain.getSaveFineParticles()) {
				cudaMemcpy(Dev_particles_fine_pos + (loop_ctr*2*Nb_Tau_p*SettingsMain.getParticlesFineNum()*2) % (2*prod_fine_dt_particles), Dev_particles_pos, 2*Nb_Tau_p*SettingsMain.getParticlesFineNum()*sizeof(double), cudaMemcpyDeviceToDevice);
			}
	    }

		// map advection
		kernel_advect_using_stream_hermite<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2,
				Dev_Psi_real, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_psi.NX, Grid_psi.NY, Grid_psi.h,
				t, dt, SettingsMain.getMapEpsilon(), SettingsMain.getTimeIntegrationNum(), SettingsMain.getMapUpdateOrderNum());
		// copy new map values onto the existing map
		cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)Dev_Temp_C1, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
		cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C2, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);

        //copy Psi to Psi_previous and Psi_previous to Psi_previous_previous
		for (int i_lagrange = 1; i_lagrange < SettingsMain.getLagrangeOrder(); i_lagrange++) {
			cudaMemcpy(Dev_Psi_real + 4*Grid_psi.N*(SettingsMain.getLagrangeOrder()-i_lagrange),
					   Dev_Psi_real + 4*Grid_psi.N*(SettingsMain.getLagrangeOrder()-i_lagrange-1), 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
		}

        /*cudaMemcpy(Dev_Psi_2048_previous_p, Dev_Psi_2048_previous, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(Dev_Psi_2048_previous, Dev_Psi_2048, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToDevice, streams[1]);*/


		/*******************************************************************
		*							 Remapping							   *
		*******************************************************************/
		
		grad_chi_min = 1;
		grad_chi_max = 1;  // not needed?
		//incompressibility check (port it on cuda)
		if(loop_ctr % 1 == 0){
			// compute gradient of map to be used for incompressibility check
			kernel_incompressibility_check<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);								// time cost		A optimiser
			// We don't need to have Dev_gradChi in memory we juste need to know if it exist a value such as : abs(this_value - 1) > inCompThreshold
			
			//cudaDeviceSynchronize();
			// compute minimum for actual check on dev, coppy to machine to get minimum from all blocks
			Dev_get_max_min<<<grad_block, grad_thread>>>(Grid_fine.N, (cufftDoubleReal*)Dev_Temp_C1, Dev_w_min, Dev_w_max);// Dev_gradChi cufftDoubleComplex cufftDoubleReal
			cudaMemcpyAsync(Host_w_min, Dev_w_min, sizeof(double)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[0]);
			cudaMemcpyAsync(Host_w_max, Dev_w_max, sizeof(double)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[1]);
			// now compute minimum in Host
			Host_get_max_min(grad_block*grad_thread, Host_w_min, Host_w_max, &grad_chi_min, &grad_chi_max);

		}
			
		//resetting map and adding to stack
		incomp_error[loop_ctr] = fmax(fabs(grad_chi_min - 1), fabs(grad_chi_max - 1));
		if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() && !SettingsMain.getSkipRemapping()) {
			if(Map_Stack.map_stack_ctr > Map_Stack.map_stack_length*Map_Stack.frac_mem_cpu_to_gpu*Map_Stack.Nb_array_RAM)
			{
				message = "Stack Saturated - Exiting"; cout<<message+"\n"; logger.push(message);
				break;
			}
			
			message = "Refining Map - ctr = " + to_str(loop_ctr) + " \t map_stack_ctr = "
					+ to_str(Map_Stack.map_stack_ctr) + " ; " + to_str(Map_Stack.stack_length_RAM) + " ; "
					+ to_str(Map_Stack.stack_length_Nb_array_RAM) + " \t gap = " + to_str(loop_ctr - old_ctr);
			cout<<message+"\n"; logger.push(message);
			old_ctr = loop_ctr;
			
			//adjusting initial conditions, compute vorticity hermite
			translate_initial_condition_through_map_stack(&Grid_fine, &Map_Stack, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real,
					cufftPlan_fine, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Temp_C1, Dev_Temp_C2);
			
			if (Map_Stack.map_stack_ctr%Map_Stack.map_stack_length == 0){
				Map_Stack.stack_length_RAM++;
				message = "stack_length_RAM = " + to_str(Map_Stack.stack_length_RAM); cout<<message+"\n"; logger.push(message);
			}
			
			if (Map_Stack.map_stack_ctr%(Map_Stack.frac_mem_cpu_to_gpu*Map_Stack.map_stack_length) == 0){
				Map_Stack.stack_length_Nb_array_RAM++;
				message = "stack_length_Nb_array_RAM = " + to_str(Map_Stack.stack_length_Nb_array_RAM); cout<<message+"\n"; logger.push(message);
			}

			Map_Stack.copy_map_to_host(Dev_ChiX, Dev_ChiY);
			Map_Stack.map_stack_ctr++;
			
			//resetting map
			kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
		}
		
		//loop counters are increased after computation
		t += dt;
		loop_ctr ++;
			
			/*******************************************************************
			*							 Save snap shot						   *
			*******************************************************************/

		if( loop_ctr % save_buffer_count == 0 )
		{
			message = "Saving Image - ctr = " + to_str(loop_ctr) + " \t save_ctr = " + to_str(save_ctr)
					+ " \t time = " + to_str(t) + " \t Compute time = " + to_str(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6);
			cout<<message+"\n"; logger.push(message);

			// save function to save variables, combined so we always save in the same way and location
			apply_map_stack_to_W_part_All(&Grid_fine, &Map_Stack, Dev_ChiX, Dev_ChiY,
					(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

			writeTimeStep(workspace, file_name, to_str(save_ctr/(double)snapshots_per_second), Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
			// compute conservation for first step
			compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);

			// save particle positions
			if (SettingsMain.getParticles()) {
				writeParticles(SettingsMain, file_name, to_str(save_ctr/(double)snapshots_per_second), Host_particles_pos, Dev_particles_pos, Tau_p, Nb_Tau_p);

			    // if wanted, save fine particles, 1 file for all positions
			    if (SettingsMain.getSaveFineParticles()) {
			    	writeFineParticles(SettingsMain, file_name, to_str(save_ctr/(double)snapshots_per_second), Host_particles_fine_pos, Dev_particles_fine_pos, Tau_p, Nb_Tau_p, prod_fine_dt_particles);
			    }
			}

		    // sample if wanted
		    if (SettingsMain.getSampleOnGrid()) {
		    	sample_compute_and_write(&Map_Stack, &Grid_sample, Host_save, Dev_save_sample,
		    			cufftPlan_sample, Dev_Temp_C1, Dev_Temp_C2,
		    			Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(),
		    			workspace, file_name, to_str(save_ctr/(double)snapshots_per_second),
		    			Mesure_sample, count_mesure);
		    }
		    count_mesure++;
			save_ctr++;

			//Laplacian_vort(&Grid_fine, Dev_W_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_lap_fine_real, Dev_lap_fine_complex, Dev_lap_fine_hat, cufftPlan_fine);

			//cudaMemcpy(Host_lap_fine, Dev_lap_fine_real, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			//writeAllRealToBinaryFile(Grid_fine.N, Host_lap_fine, simulationName, "vorticity_fine_lagrangian/w_lagr_" + ss.str());


			 //Laplacian initial

			if (use_set_grid == 1) {
//					Laplacian_vort(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_2048, Dev_lap_fine_2048_real, Dev_lap_fine_2048_complex, Dev_lap_fine_2048_hat, cufftPlan_2048);
//					cudaMemcpy(Host_lap_fine_2048, Dev_lap_fine_2048_real, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//					cudaDeviceSynchronize();
//					writeAllRealToBinaryFile(Grid_2048.N, Host_lap_fine_2048, workspace, file_name, "vorticity_fine_lagrangian/w_lagr_" + ss.str());
			}

		}
		
		int error = cudaGetLastError();
		//cudaError_t err = cudaGetLastError();
		//if (err != cudaSuccess){
		//	  printf("%s\n", cudaGetErrorString(err));
		//}
		if(error != 0)
		{
			message = "Exited early; Last Cuda Error : " + to_str(error); cout<<message+"\n"; logger.push(message);
			exit(0);
			break;
		}

		// save timing at last part of a step to take everything into account
		{
			auto step = std::chrono::high_resolution_clock::now();
			double diff = std::chrono::duration_cast<std::chrono::microseconds>(step - begin).count()/1e6;
			time_values[loop_ctr] = diff; // loop_ctr was already increased but first entry is init time
		}
		message = "Step : " + to_str(loop_ctr) + " ,\t Incomp Error : " + to_str(incomp_error[loop_ctr-1])
				+ " ,\t Time : " + to_str(time_values[loop_ctr]); cout<<message+"\n"; logger.push(message);
	}
	
	
	
	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/
	
	// save function to save variables, combined so we always save in the same way and location, last step so we can actually modify the condition
	apply_map_stack_to_W_part_All(&Grid_fine, &Map_Stack, Dev_ChiX, Dev_ChiY,
			(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());

	writeTimeStep(workspace, file_name, "final", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
	// compute conservation
	compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);

    // sample if wanted
    if (SettingsMain.getSampleOnGrid()) {
    	sample_compute_and_write(&Map_Stack, &Grid_sample, Host_save, Dev_save_sample,
    			cufftPlan_sample, Dev_Temp_C1, Dev_Temp_C2,
    			Dev_ChiX, Dev_ChiY, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum(),
    			workspace, file_name, "final",
    			Mesure_sample, count_mesure);
    }
	count_mesure++;

	if (SettingsMain.getParticles()) {
		writeParticles(SettingsMain, file_name, "final", Host_particles_pos, Dev_particles_pos, Tau_p, Nb_Tau_p);
	}

	// save all conservation data
	writeAllRealToBinaryFile(3*mes_size, Mesure, workspace, file_name, "/Mesure");
	writeAllRealToBinaryFile(3*mes_size, Mesure_fine, workspace, file_name, "/Mesure_fine");
	writeAllRealToBinaryFile(3*mes_size, Mesure_sample, workspace, file_name, "/Mesure_"+to_str(Grid_sample.NX));

    // save imcomp error
	writeAllRealToBinaryFile(iterMax, incomp_error, workspace, file_name, "/Incompressibility_check");
	
	
	/*******************************************************************
	*					  Zoom on the last frame					   *
	*******************************************************************/
	
//	Zoom(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack,
//			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//			Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM,
//			Dev_W_fine, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, simulationName, LX);
	
	
	/*******************************************************************
	*						 Finalisation							   *
	*******************************************************************/

	/*
	kernel_compare_vorticity_with_initial<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Dev_W_fine, map_stack_ctr, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);
	cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	get_max_min(&Grid_fine, Host_W_fine, &w_min, &w_max);
	cout<<fabs(w_min)<<endl<<fabs(w_max)<<endl;
	//writeRealToImage(&Grid_fine, Host_W_fine, simulationName + "/w_bmp/Error" , w_min, w_max, JET, true);	
	*/
	
	double maxError = fabs(w_max);
	if(fabs(w_min) > fabs(w_max))
		maxError = fabs(w_min);
	
	// i don't know if this is important, i don't think so
//	char buffer[50];
//	int cudaError = cudaGetLastError();
//	ofstream errorLogFile("data/errorLog.csv", ios::out | ios::app);
//	sprintf(buffer, "%e", maxError);
//	errorLogFile<<NX_coarse<<", "<<fabs(dt)<<", "<<tf<<", "<<buffer<<","<<cudaError<<endl;
//	errorLogFile.close();
	
	
	
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
	
	cudaFree(Dev_w_min);
	cudaFree(Dev_w_max);

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
	writeAllRealToBinaryFile(iterMax+2, time_values, workspace, file_name, "/Timing_Values");

	message = "Finished - Last Cuda Error = " + to_str(cudaGetLastError()) + " , Time = " + to_str(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin).count()/1e6); cout<<message+"\n"; logger.push(message);
}


/*******************************************************************
*							 Remapping							   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_fine, MapStack *Map_Stack, double *Dev_ChiX, double *Dev_ChiY,
		double *W_H_real, cufftHandle cufftPlan_fine, double *bounds, double *W_initial, int simulation_num_c,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2)
{
	
	// Vorticity on coarse grid to vorticity on fine grid with a very long command, use Dev_Temp_C1 for W_fine
	apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
			(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, bounds, W_initial, simulation_num_c);

	//kernel_apply_map_stack_to_W<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, W_real, stack_length, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
	k_real_to_comp<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, Grid_fine->NX, Grid_fine->NY);

	cufftExecZ2Z(cufftPlan_fine, Dev_Temp_C2, Dev_Temp_C1, CUFFT_FORWARD);
	k_normalize<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Temp_C1, Grid_fine->NX, Grid_fine->NY);

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3)
//	k_fft_cut_off_scale<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Temp_C1, Grid_fine->NX, (double)(Grid_fine->NX)/3.0);
	
	// form hermite formulation
	fourier_hermite(Grid_fine, Dev_Temp_C1, W_H_real, Dev_Temp_C2, cufftPlan_fine);
}


/*******************************************************************
*						 Computation of Psi						   *
*******************************************************************/

// upsample psi by doing zero padding vorticity in fourier space from coarse grid to psi grid
void evaluate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_psi, double *Dev_ChiX, double *Dev_ChiY,
		double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_psi,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, int molly_stencil, double freq_cut_psi)
{

	// apply map to w and sample using mollifier
	kernel_apply_map_and_sample_from_hermite<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, W_real, Dev_W_H_fine_real, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, molly_stencil);

	// forward fft
	k_real_to_comp<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Temp_C1, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Grid_coarse->NX, Grid_coarse->NY);

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3) and compute smooth W
	// use Psi grid here for intermediate storage
//	k_fft_cut_off_scale<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Grid_coarse->NX, (double)(Grid_psi->NX)/3.0);


	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C2, (cufftDoubleComplex*)Psi_real, CUFFT_INVERSE);
	comp_to_real((cufftDoubleComplex*)Psi_real, W_real, Grid_coarse->N);

	// zero padding by moving all entries and creating a middle zone with zeros
	// initialize zeros for padding for trash variable to be used
	cudaMemset(Dev_Temp_C1, 0, Grid_psi->sizeNComplex);
	k_fft_grid_add<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse->NX, Grid_psi->NX);

	// cut high frequencies in fourier space, however not that much happens after zero move add from coarse grid
	k_fft_cut_off_scale<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Grid_psi->NX, freq_cut_psi);

	// Forming Psi hermite now on psi grid
	k_fft_iLap<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);												// Inverse laplacian in Fourier space
	fourier_hermite(Grid_psi, Dev_Temp_C2, Psi_real, Dev_Temp_C1, cufftPlan_psi);
}


// compute vorticity directly on psi grid and keep it for psi, only downsample for w_real to coarse grid
void evaluate_stream_hermite_2(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_psi, double *Dev_ChiX, double *Dev_ChiY,
		double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_psi,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, int molly_stencil, double freq_cut_psi, double *Host_debug)
{

	// apply map to w and sample using mollifier, since we use psi grid, it is firstly transcribed into psi just cause we can use the grid
	kernel_apply_map_and_sample_from_hermite<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Psi_real, Dev_W_H_fine_real, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_psi->NX, Grid_psi->NY, Grid_psi->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, molly_stencil);

	// forward fft on psi grid
	k_real_to_comp<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Psi_real, Dev_Temp_C1, Grid_psi->NX, Grid_psi->NY);
	cufftExecZ2Z(cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Grid_psi->NX, Grid_psi->NY);

	// cut_off frequencies at N_psi/3 for turbulence (effectively 2/3) and compute smooth W
	// use Psi grid here for intermediate storage
//	k_fft_cut_off_scale<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Grid_psi->NX, (double)(Grid_psi->NX)/3.0);

	k_fft_grid_remove<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse->NX, Grid_psi->NX);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, (cufftDoubleComplex*)Psi_real, CUFFT_INVERSE);
	comp_to_real((cufftDoubleComplex*)Psi_real, W_real, Grid_coarse->N);

	// cut high frequencies in fourier space
	k_fft_cut_off_scale<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Grid_psi->NX, freq_cut_psi);

	// Forming Psi hermite on psi grid
	k_fft_iLap<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);												// Inverse laplacian in Fourier space
	fourier_hermite(Grid_psi, Dev_Temp_C1, Psi_real, Dev_Temp_C2, cufftPlan_psi);
}
// debugging lines, could be needed here to check psi
//	cudaMemcpy(Host_Debug, Psi_real, 4*Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(4*Grid_psi->N, Host_Debug, "psi_debug_4_nodes_C512_F2048_t64_T1", "Debug_2");


// sample psi on a fixed grid with vorticity known
void psi_upsampling(TCudaGrid2D *Grid, double *Dev_W, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, double *Dev_Psi, cufftHandle cufftPlan){
	k_real_to_comp<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_W, Dev_Temp_C1, Grid->NX, Grid->NY);

	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_FORWARD);
	k_normalize<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C2, Grid->NX, Grid->NY);

	// Forming Psi hermite
	k_fft_iLap<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid->NX, Grid->NY, Grid->h);													// Inverse laplacian in Fourier space
	fourier_hermite(Grid, Dev_Temp_C1, Dev_Psi, Dev_Temp_C2, cufftPlan);
}


// compute hermite with derivatives in fourier space, uniform helper function fitted for all grids to utilize only two trash variables
// input is given in first trash variable
void fourier_hermite(TCudaGrid2D *Grid, cufftDoubleComplex *Dev_Temp_C1, double *Dev_Output, cufftDoubleComplex *Dev_Temp_C2, cufftHandle cufftPlan) {
	// dy and dxdy derivates are stored in later parts of output array, we can therefore use the first half as a trash variable
	// start with dy derivative and store in position 3/4
	k_fft_dy<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid->NX, Grid->NY, Grid->h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan, Dev_Temp_C2, (cufftDoubleComplex*)Dev_Output, CUFFT_INVERSE);
	comp_to_real((cufftDoubleComplex*)Dev_Output, Dev_Output + 2*Grid->N, Grid->N);

	// reuse values from Trash_C2 and create dxdy, store in position 4/4
	k_fft_dx<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C2, (cufftDoubleComplex*)Dev_Output, Grid->NX, Grid->NY, Grid->h);
	cufftExecZ2Z(cufftPlan, (cufftDoubleComplex*)Dev_Output, Dev_Temp_C2, CUFFT_INVERSE);
	comp_to_real(Dev_Temp_C2, Dev_Output + 3*Grid->N, Grid->N);

	// we can go back to Trash_C1 and store the other two values normally, first normal values
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);
	comp_to_real(Dev_Temp_C2, Dev_Output, Grid->N);

	// now compute dx derivative and store it
	k_fft_dx<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid->NX, Grid->NY, Grid->h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan, Dev_Temp_C2, Dev_Temp_C1, CUFFT_INVERSE);
	comp_to_real(Dev_Temp_C1, Dev_Output + Grid->N, Grid->N);
}


/*******************************************************************
*		 Computation of Global conservation values				   *
*******************************************************************/

void compute_conservation_targets(TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi,
		double *Host_save, double *Dev_Psi, double *Dev_W_coarse, double *Dev_W_H_fine,
		cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_fine,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2,
		double *Mesure, double *Mesure_fine, int count_mesure)
{
	#ifndef sm_50
		// coarse grid
		Compute_Energy<<<Grid_psi->blocksPerGrid,Grid_psi->threadsPerBlock>>>(&Mesure[3*count_mesure], Dev_Psi, Grid_psi->N, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);
		Compute_Enstrophy<<<Grid_coarse->blocksPerGrid,Grid_coarse->threadsPerBlock>>>(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse->N, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);
		// fine grid, no energy because we do not have velocity on fine grid
		Compute_Enstrophy<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(&Mesure_fine[1 + 3*count_mesure], Dev_W_H_fine, Grid_fine->N, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
	#else
		// coarse grid
		cudaMemcpy(Host_save, Dev_Psi, 4*Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
		Compute_Energy_Host(&Mesure[3*count_mesure], Host_save, Grid_psi->N, Grid_psi->h);
		cudaMemcpy(Host_save, Dev_W_coarse, Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
		Compute_Enstrophy_Host(&Mesure[1 + 3*count_mesure], Host_save, Grid_coarse->N, Grid_coarse->h);
		// fine grid, no energy because we do not have velocity on fine grid
		cudaMemcpy(Host_save, Dev_W_H_fine, Grid_fine->sizeNReal, cudaMemcpyDeviceToHost);
		Compute_Enstrophy_Host(&Mesure_fine[1 + 3*count_mesure], Host_save, Grid_fine->N, Grid_fine->h);
	#endif
	// palinstrophy is computed on Host, fine first because vorticity is saved in temporal array
	Compute_Palinstrophy_fourier(Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_H_fine, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_fine);
	Compute_Palinstrophy_fourier(Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_coarse);
}


/*******************************************************************
*		 Sample on a specific grid and save everything	           *
*******************************************************************/

void sample_compute_and_write(MapStack *Map_Stack, TCudaGrid2D *Grid_sample, double *Host_sample, double *Dev_sample,
		cufftHandle cufftPlan_sample, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2,
		double *Dev_ChiX, double*Dev_ChiY, double *bounds, double *W_initial, int simulation_num_c,
		string workspace, string sim_name, string i_num,
		double *Mesure_sample, int count_mesure) {

	// begin with vorticity
	apply_map_stack_to_W_part_All(Grid_sample, Map_Stack, Dev_ChiX, Dev_ChiY,
			Dev_sample, (cufftDoubleReal*)Dev_Temp_C1, bounds, W_initial, simulation_num_c);
	writeTimeVariable(workspace, sim_name, "Vorticity_W_"+to_str(Grid_sample->NX), i_num, Host_sample, Dev_sample, Grid_sample->sizeNReal, Grid_sample->N);

	// compute enstrophy and palinstrophy, data already on host
	#ifndef sm_50
		Compute_Enstrophy<<<Grid_sample->blocksPerGrid,Grid_sample->threadsPerBlock>>>(&Mesure_sample[1 + 3*count_mesure], Dev_sample, Grid_sample->N, Grid_sample->NX, Grid_sample->NY, Grid_sample->h);
	#else
		Compute_Enstrophy_Host(&Mesure_sample[1 + 3*count_mesure], Host_sample, Grid_sample->N, Grid_sample->h);
	#endif
	Compute_Palinstrophy_fourier(Grid_sample, &Mesure_sample[2 + 3*count_mesure], Dev_sample, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_sample);

	// reuse sampled vorticity to compute psi
	psi_upsampling(Grid_sample, Dev_sample, Dev_Temp_C1, Dev_Temp_C2, Dev_sample, cufftPlan_sample);
	writeTimeVariable(workspace, sim_name, "Stream_Function_Psi_"+to_str(Grid_sample->NX), i_num, Host_sample, Dev_sample, 4*Grid_sample->sizeNReal, 4*Grid_sample->N);

	// compute energy, data already on host
	#ifndef sm_50
		Compute_Energy<<<Grid_sample->blocksPerGrid,Grid_sample->threadsPerBlock>>>(&Mesure_sample[3*count_mesure], Dev_sample, Grid_sample->N, Grid_sample->NX, Grid_sample->NY, Grid_sample->h);
	#else
		Compute_Energy_Host(&Mesure_sample[3*count_mesure], Host_sample, Grid_sample->N, Grid_sample->h);
	#endif

	// map
	k_sample<<<Grid_sample->blocksPerGrid,Grid_sample->threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_sample, Dev_sample + Grid_sample->N, Map_Stack->Grid->NX, Map_Stack->Grid->NY, Map_Stack->Grid->h, Grid_sample->NX, Grid_sample->NY, Grid_sample->h);
	writeTimeVariable(workspace, sim_name, "Map_ChiX_"+to_str(Grid_sample->NX), i_num, Host_sample, Dev_sample, Grid_sample->sizeNReal, Grid_sample->N);
	writeTimeVariable(workspace, sim_name, "Map_ChiY_"+to_str(Grid_sample->NX), i_num, Host_sample, Dev_sample + Grid_sample->N, Grid_sample->sizeNReal, Grid_sample->N);
}



// not finalized yet, i wanted to use this to upsample psi and the map, however, I would need a third temporal variable for this, i'm thinking of using W_H_fine and temporali copy it onto the host
// however, I can already tell, that I want to use this to be able to save on a fixed grid, there we could initialize a temporal variable for that
///*******************************************************************
//*		 			Prepare save of timestep
//*******************************************************************/
//void prepare_write_timestep(string workspace, string sim_name, string i_num, double *Host_save, double *Dev_W_coarse, cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, double *Dev_Psi_real, double *Dev_ChiX, double *Dev_ChiY, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi) {
//	// save vorticity coarse
//	writeTimeVariable(workspace, sim_name, "Vorticity_W_coarse", i_num, Host_save, Dev_W_coarse, &Grid_coarse);
//
//	// compute vorticity on fine grid
//	apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//			(cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
//			Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, bounds, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
//	// save vorticity fine
//	writeTimeVariable(workspace, sim_name, "Vorticity_W_fine", i_num, Host_save, (cufftDoubleReal*)Dev_Temp_C1, &Grid_fine);
//
//	// save psi psi
//	writeTimeVariable(workspace, sim_name, "Vorticity_W_fine", i_num, Host_save, (cufftDoubleReal*)Dev_Temp_C1, &Grid_fine);
//
//	// save vort and psi on coarse and fine grid and map on coarse grid
//	writeTimeStep(workspace, file_name, to_str(save_ctr), Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
//
//	// save map
//	writeTimeVariable(workspace, sim_name, "Map_ChiX_coarse", i_num, Host_save, Dev_ChiX, &Grid_coarse);
//	writeTimeVariable(workspace, sim_name, "Map_ChiX_coarse", i_num, Host_save, Dev_ChiY, &Grid_coarse);
//}



/*******************************************************************
*							   Zoom								   *
*******************************************************************/


void Zoom(TCudaGrid2D *Grid_fine, MapStack *Map_Stack, double *Dev_ChiX, double *Dev_ChiY, double *W_real,
		cufftHandle cufftPlan_fine, double *W_initial, cufftDoubleComplex *Dev_Temp,
		string workspace, string simulationName, int simulation_num, double L)
{
	double *ws;
	ws = new double[Grid_fine->N];
	int save_ctr = 0;

	double xCenter = 0.54;
	double yCenter = 0.51;
	double width = 0.5;

	double xMin = xCenter - width/2;
	double xMax = xMin + width;
	double yMin = yCenter - width/2;
	double yMax = yMin + width;

	std::ostringstream ss;
	ss<<save_ctr;


	//save zooming effects
	for(int zoom_ctr = 0; zoom_ctr<10; zoom_ctr++){

		width *=  0.5;//0.99
		xMin = xCenter - width/2;
		xMax = xMin + width;
		yMin = yCenter - width/2;
		yMax = yMin + width;

		double bounds[4] = {xMin, xMax, yMin, yMax};


		//kernel_apply_map_stack_to_W_custom<<<Gsf->blocksPerGrid, Gsf->threadsPerBlock>>>(devChiX_stack, devChiY_stack, devChiX, devChiY, devWs, stack_map_passed, Gc->NX, Gc->NY, Gc->h, Gsf->NX, Gsf->NY, Gsf->h, xMin*L, xMax*L, yMin*L, yMax*L, W_initial);
		apply_map_stack_to_W_part_All(Grid_fine, Map_Stack, Dev_ChiX, Dev_ChiY,
				W_real, (cufftDoubleReal*)Dev_Temp, bounds, W_initial, simulation_num);

		cudaMemcpy(ws, W_real, Grid_fine->sizeNReal, cudaMemcpyDeviceToHost);

		std::ostringstream ss2;
		ss2<<zoom_ctr;

		writeAllRealToBinaryFile(Grid_fine->N, ws, simulationName, workspace, "zoom_" + ss2.str());
	}

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
	
	
	Zoom(simulationName, LX, &Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Dev_W_fine, map_stack_ctr);	
	
	
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



























