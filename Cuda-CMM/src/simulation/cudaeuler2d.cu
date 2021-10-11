#include "cudaeuler2d.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

struct stat st = {0};



//void cuda_euler_2d(string intial_condition, int grid_scale, int fine_grid_scale, string time_integration, string map_update_order, int molly_stencil, double final_time_override, double time_step_factor)
void cuda_euler_2d(SettingsCMM SettingsMain)
{
	
	// start clock as first thing to measure initializations too
	clock_t begin = clock();

	/*******************************************************************
	*						 	 Constants							   *
	*******************************************************************/
	
	double LX = twoPI;													// domain length
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
	int Nb_array_RAM = SettingsMain.getNbArrayRam();					// fixed for four different stacks
	int use_set_grid = 0;  // change later, use 2048 grid thingys or not

	// compute dt, for grid settings we have to use max in case we want to differ NX and NY
	if (SettingsMain.getSetDtBySteps()) dt = 1.0 / steps_per_sec;  // direct setting of timestep
	else dt = 1.0 / std::max(NX_coarse, NY_coarse) / grid_by_time;  // setting of timestep by grid and factor
	
	#ifdef DISCRET
		grid_by_time = 8.0;
		snapshots_per_second = 1;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 100;	
	#endif
	#ifndef DISCRET
		double *Dev_W_H_initial;
		cudaMalloc((void**)&Dev_W_H_initial, 8);
	#endif
	
	
	//shared parameters
	iterMax = ceil(tf / dt);
	save_buffer_count = 1.0/snapshots_per_second/dt;//(NX_coarse * grid_by_time) / snapshots_per_second;//	tmp/snapshots_per_second;//(NX_coarse * grid_by_time) / snapshots_per_second;							// the denominator is snapshots per second
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
	*																   *
	*******************************************************************/
	
	TCudaGrid2D Grid_coarse(NX_coarse, NY_coarse, LX);
	TCudaGrid2D Grid_fine(NX_fine, NY_fine, LX);
	TCudaGrid2D Grid_psi(NX_psi, NY_psi, LX);
	// TCudaGrid2D Grid_NDFT(128, 128, LX);
	
	
	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufftPlan_coarse, cufftPlan_fine, cufftPlan_psi;
	cufftPlan2d(&cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z);
	cufftPlan2d(&cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z);
	cufftPlan2d(&cufftPlan_psi, Grid_psi.NX, Grid_psi.NY, CUFFT_Z2Z);
	
	
	/*******************************************************************
	*							Trash variable
	*          Used for temporary copying and storing of data
	*                       in various locations
	*  casting (cufftDoubleReal*) makes them usable for double arrays
	*******************************************************************/
	
	// set after largest grid, checks are for failsave in case one chooses weird grid
	long int size_max_c = std::max(Grid_fine.sizeNComplex, Grid_psi.sizeNComplex);
	size_max_c = std::max(size_max_c, 4*Grid_coarse.sizeNComplex);
	// for now three thrash variables are needed for cufft with hermites
	cufftDoubleComplex *Dev_Temp_C1, *Dev_Temp_C2;
	cudaMalloc((void**)&Dev_Temp_C1, size_max_c);
	cudaMalloc((void**)&Dev_Temp_C2, size_max_c);
	
	// we actually only need one host array, as we always just copy from and to this and never really read files
	long int size_max_r = std::max(Grid_fine.N, 4*Grid_psi.N);
	size_max_r = std::max(size_max_r, 4*Grid_coarse.N);
	double *Host_save;
	Host_save = new double[size_max_r];

	
	/*******************************************************************
	*								Test NDFT						   *
	*******************************************************************/
	
	/*
	printf("NDFT\n");
	
	int Np_particles = 16384;
	int iNDFT_block, iNDFT_thread = 256;
	iNDFT_block = Np_particles/256;
	int *f_k, *Dev_f_k;
	double *x_1_n, *x_2_n, *p_n, *Dev_x_1_n, *Dev_x_2_n, *Dev_p_n, *X_k_bis;
	cufftDoubleComplex *X_k, *Dev_X_k, *Dev_X_k_derivative;
	
	x_1_n = new double[Np_particles];
	x_2_n = new double[Np_particles];
	p_n = new double[2*Np_particles];
	f_k = new int[Grid_NDFT.NX];
	X_k = new cufftDoubleComplex[Grid_NDFT.N];
	X_k_bis = new double[2*Grid_NDFT.N];
	cudaMalloc((void**)&Dev_x_1_n, sizeof(double)*Np_particles);
	cudaMalloc((void**)&Dev_x_2_n, sizeof(double)*Np_particles);
	cudaMalloc((void**)&Dev_p_n, sizeof(double)*2*Np_particles);
	cudaMalloc((void**)&Dev_f_k, sizeof(int)*Np_particles);
	cudaMalloc((void**)&Dev_X_k, Grid_NDFT.sizeNComplex);
	cudaMalloc((void**)&Dev_X_k_derivative, Grid_NDFT.sizeNComplex);
	
	readRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x1.data");
	readRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x2.data");
	readRealToBinaryAnyFile(2*Np_particles, p_n, "src/Initial_W_discret/p.data");
	
	for(int i = 0; i < Grid_NDFT.NX; i+=1)
		f_k[i] = i;
	
	cudaMemcpy(Dev_x_1_n, x_1_n, sizeof(double)*Np_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_x_2_n, x_2_n, sizeof(double)*Np_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_p_n, p_n, sizeof(double)*2*Np_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_f_k, f_k, sizeof(int)*Grid_NDFT.NX, cudaMemcpyHostToDevice);
	
	printf("NDFT v_x\n");
	NDFT_2D<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX, Np_particles);
	printf("iNDFT v_x\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_1_n, Dev_x_1_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x_1_ifft.data");
	
	printf("k_fft_dx\n");
	k_fft_dx<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
	printf("iNDFT v_x/dx\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k_derivative, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_1_n, Dev_x_1_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x_1_dx_ifft.data");
	
	cudaMemcpy(X_k, Dev_X_k_derivative, Grid_NDFT.sizeNComplex, cudaMemcpyDeviceToHost);
	printf("%lf %lf %lf\n", X_k[0].x, X_k[1].x, X_k[Grid_NDFT.N-1].x);
	//writeRealToBinaryAnyFile(2*Np_particles, X_k, "src/Initial_W_discret/X_k.data");
	
	for(int i = 0; i < Grid_NDFT.N; i+=1){
		X_k_bis[2*i] 	= 	X_k[i].x;
		X_k_bis[2*i+1] 	= 	X_k[i].y;
	}
	writeRealToBinaryAnyFile(2*Grid_NDFT.N, X_k_bis, "src/Initial_W_discret/X_k.data");
	
	
	printf("NDFT v_y\n");
	NDFT_2D<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX, Np_particles);
	printf("iNDFT v_y\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_2_n, Dev_x_2_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x_2_ifft.data");
	
	printf("k_fft_dy\n");
	k_fft_dy<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
	printf("iNDFT v_y/dy\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k_derivative, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_2_n, Dev_x_2_n, sizeof(double)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x_2_dy_ifft.data");
	
	printf("Fini NDFT\n");
	*/
	
	/*******************************************************************
	*							  Chi								   *
	* 	Chi is an array that contains Chi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative   					       *
	* 																   *
	*******************************************************************/
	
	double *Dev_ChiX, *Dev_ChiY;
	
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	
	
	/*******************************************************************
	*					       Chi_stack							   *
	* 	We need to save the variable Chi to be able to make	the        *
	* 	remapping or the zoom				   					       *
	* 																   *
	*******************************************************************/
	
	double *Host_ChiX_stack_RAM_0, *Host_ChiY_stack_RAM_0, *Host_ChiX_stack_RAM_1, *Host_ChiY_stack_RAM_1, *Host_ChiX_stack_RAM_2, *Host_ChiY_stack_RAM_2, *Host_ChiX_stack_RAM_3, *Host_ChiY_stack_RAM_3, *Dev_ChiX_stack, *Dev_ChiY_stack;
	
	cudaMalloc((void **) &Dev_ChiX_stack, map_stack_length * 4*Grid_coarse.sizeNReal);	
	cudaMalloc((void **) &Dev_ChiY_stack, map_stack_length * 4*Grid_coarse.sizeNReal);
	int map_stack_ctr = 0;
	message = "Map Stack Initialized"; cout<<message+"\n"; logger.push(message);
	
	int stack_length_RAM = -1;
	int stack_length_Nb_array_RAM = -1;
	Host_ChiX_stack_RAM_0 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_0 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiX_stack_RAM_1 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_1 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiX_stack_RAM_2 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_2 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiX_stack_RAM_3 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_3 = new double[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	
	
	/*******************************************************************
	*					       Vorticity							   *
	* 	We need to have different variable version. coarse/fine,       *
	* 	real/complex/hat and an array that contains NE, SE, SW, NW	   *
	* 																   *
	*******************************************************************/
	
	double *Dev_W_coarse, *Dev_W_H_fine_real;
	cudaMalloc((void**)&Dev_W_coarse, Grid_coarse.sizeNReal);
	//vorticity hermite
	cudaMalloc((void**)&Dev_W_H_fine_real, 4*Grid_fine.sizeNReal);
	
	
	/*******************************************************************
	*							DISCRET								   *
	*******************************************************************/
	
	#ifdef DISCRET
		
		double *Host_W_initial, *Dev_W_H_initial;
		Host_W_initial = new double[Grid_fine.N];
		cudaMalloc((void**)&Dev_W_H_initial, 4*Grid_fine.sizeNReal);
		
		std::ostringstream fine_grid_scale_nb;
		fine_grid_scale_nb<<fine_grid_scale;
		
		readRealToBinaryAnyFile(Grid_fine.N, Host_W_initial, "src/Initial_W_discret/file2D_" + fine_grid_scale_nb.str() + ".bin");
		
		cudaMemcpy(Dev_W_fine, Host_W_initial, Grid_fine.sizeNReal, cudaMemcpyHostToDevice);
		
		k_real_to_comp<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_W_fine, Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Temp_C1, CUFFT_FORWARD);
		k_normalize<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Temp_C1, Grid_fine.NX, Grid_fine.NY);
		
		// Hermite vorticity array : [vorticity, x-derivative, y-derivative, xy-derivative]
		cudaMemcpy(Dev_W_H_initial, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToDevice);
		
		k_fft_dy<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Temp_C1, Dev_Hat_fine_bis, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);													// y-derivative of the vorticity in Fourier space
		cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine_bis, Dev_Complex_fine, CUFFT_INVERSE);
		k_comp_to_real  <<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Dev_W_H_initial[2*Grid_fine.N], Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		
		k_fft_dx<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Temp_C1, Dev_Hat_fine_bis, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);													// x-derivative of the vorticity in Fourier space
		cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine_bis, Dev_Complex_fine, CUFFT_INVERSE);
		k_comp_to_real  <<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Dev_W_H_initial[Grid_fine.N], Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		
		k_fft_dy<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Hat_fine_bis, Dev_Temp_C1, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);													// y-derivative of x-derivative of of the vorticity in Fourier space
		cufftExecZ2Z(cufftPlan_fine, Dev_Temp_C1, Dev_Complex_fine, CUFFT_INVERSE);
		k_comp_to_real  <<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Dev_W_H_initial[3*Grid_fine.N], Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		
		delete [] Host_W_initial;
		
		cout<<cudaGetErrorName (cudaGetLastError());
		printf("\n");
		
	#endif
	
	/*******************************************************************
	*							  Psi								   *
	* 	Psi is an array that contains Psi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative 							   *
	* 																   *
	*******************************************************************/
	
	//stream hermite on coarse computational grid
	double *Dev_Psi_real, *Dev_Psi_real_previous, *Dev_Psi_real_previous_p;

	cudaMalloc((void**) &Dev_Psi_real, 4*Grid_psi.sizeNReal);
	cudaMalloc((void**) &Dev_Psi_real_previous, 4*Grid_psi.sizeNReal);
	cudaMalloc((void**) &Dev_Psi_real_previous_p, 4*Grid_psi.sizeNReal);

	
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
	
	
	/*******************************************************************
	*							 Particles							   *
	*******************************************************************/
	// all variables have to be defined outside
	int Nb_particles, particle_thread, particle_block;
	double *Host_particles_pos ,*Dev_particles_pos;  // position of particles
	double *Host_particles_vel ,*Dev_particles_vel;  // velocity of particles
	double *Dev_particles_vel_p, *Dev_particles_vel_p_p;  // previous time step velocities needed for time stepping
	curandGenerator_t prng;
	int Nb_fine_dt_particles, freq_fine_dt_particles, prod_fine_dt_particles, taup_fine_particles;
	double *Dev_particles_pos_fine_dt, *Host_particles_pos_fine_dt;
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
		cudaMalloc((void**) &Dev_particles_vel_p, 2*Nb_particles*Nb_Tau_p*sizeof(double));
		cudaMalloc((void**) &Dev_particles_vel_p_p, 2*Nb_particles*Nb_Tau_p*sizeof(double));

		// create initial positions from random distribution
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p);

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
			cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		// particles where every time step the position will be saved
		Nb_fine_dt_particles = 1000;
		Nb_fine_dt_particles = std::min(Nb_fine_dt_particles, Nb_particles);  // make sure we don't exceed particle number
		freq_fine_dt_particles = save_buffer_count; // redundant
		prod_fine_dt_particles = Nb_fine_dt_particles * freq_fine_dt_particles;
		taup_fine_particles = 0;  // little hack to be able to safe fine instances of inertial particles

		Host_particles_pos_fine_dt = new double[2*prod_fine_dt_particles];
		cudaMalloc((void**) &Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(double));

		message = "Number of particles = " + to_str(Nb_particles); cout<<message+"\n"; logger.push(message);
	}


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	int count_mesure = 0;
	int mes_size = tf*snapshots_per_second + 2;  // add initial and last step
    double *Mesure;
    double *Mesure_fine;
	cudaMallocManaged(&Mesure, 3*mes_size*sizeof(double));
	cudaMallocManaged(&Mesure_fine, 3*mes_size*sizeof(double));

    double incomp_error [iterMax];  // save incompressibility error for investigations
    double time_values [iterMax+2];  // save timing for investigations


    // File organization : Might be moved
    std::string fi, element[6] = {"particles", "vorticity_coarse", "vorticity_fine", "stream_function", "vorticity_fine_lagrangian", "map_coarse"};
    for(int i = 0; i<6; i+=1){
        fi = workspace + "data/" + file_name + "/all_save_data/" + element[i];
        mkdir(fi.c_str(), 0700);
    }

    if (SettingsMain.getParticles()) {
        fi = workspace + "data/" + file_name + "/all_save_data/particles/fluid";
        mkdir(fi.c_str(), 0700);
        for(int i = 1; i<Nb_Tau_p; i+=1){
            fi = workspace + "data/" + file_name + "/all_save_data/particles/" + std::to_string(Tau_p[i]).substr(0, std::to_string(Tau_p[i]).find(".") + 3+ 1);
            mkdir(fi.c_str(), 0700);
        }
	}


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
	*******************************************************************/
	
	
	TCudaGrid2D Grid_2048(1024, 1024, LX);
	
	double *Host_2048_4;
	double *Dev_ChiX_2048, *Dev_ChiY_2048, *Dev_W_2048;
	cufftHandle cufftPlan_2048;
	cufftDoubleComplex *Dev_Complex_fine_2048, *Dev_Hat_fine_2048, *Dev_Hat_fine_bis_2048;
    double *Dev_Psi_2048, *Dev_Psi_2048_previous, *Dev_Psi_2048_previous_p;
    double *Host_lap_fine_2048, *Dev_lap_fine_2048_real;
    cufftDoubleComplex *Dev_lap_fine_2048_complex, *Dev_lap_fine_2048_hat;

    if (use_set_grid == 1) {
	
		Host_2048_4 = new double[4*Grid_2048.N];
		cudaMalloc((void**)&Dev_ChiX_2048, Grid_2048.sizeNReal);
		cudaMalloc((void**)&Dev_ChiY_2048, Grid_2048.sizeNReal);

		cudaMalloc((void**)&Dev_W_2048, Grid_2048.sizeNReal);

		cufftPlan2d(&cufftPlan_2048, Grid_2048.NX, Grid_2048.NY, CUFFT_Z2Z);


		cudaMalloc((void**)&Dev_Complex_fine_2048, Grid_2048.sizeNComplex);
		cudaMalloc((void**)&Dev_Hat_fine_2048, Grid_2048.sizeNComplex);
		cudaMalloc((void**)&Dev_Hat_fine_bis_2048, Grid_2048.sizeNComplex);


		cudaMalloc((void**) &Dev_Psi_2048, 4*Grid_2048.sizeNReal);
		cudaMalloc((void**) &Dev_Psi_2048_previous, 4*Grid_2048.sizeNReal);
		cudaMalloc((void**) &Dev_Psi_2048_previous_p, 4*Grid_2048.sizeNReal);
	
		// Laplacian
	
		Host_lap_fine_2048 = new double[Grid_2048.N];
	
		cudaMalloc((void**)&Dev_lap_fine_2048_real, Grid_2048.sizeNReal);
		cudaMalloc((void**)&Dev_lap_fine_2048_complex, Grid_2048.sizeNComplex);
		cudaMalloc((void**)&Dev_lap_fine_2048_hat, Grid_2048.sizeNComplex);
    
    }





	/*******************************************************************
	*						   Initialization						   *
	*******************************************************************/
		
	//initialization of flow map as normal grid
	kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	
	//setting initial conditions for vorticity by translating with initial grid
	translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack,
			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
			Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
			Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Temp_C1, Dev_Temp_C2);
	
	// we need psi, psi_p and psi_p_p for particle initialization
	// compute psi and store it in psi_p
	// compute stream hermite from vorticity, we have two different versions avaiable
	if (SettingsMain.getUpsampleVersion() == 0) {
		evaluate_stream_hermite(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi());
	}
	else {
		evaluate_stream_hermite_2(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Temp_C1, Dev_Temp_C2, SettingsMain.getMollyStencil(), SettingsMain.getFreqCutPsi(), Host_save);
	}
    //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
    // set psi_p_p and psi_p after psi
    cudaMemcpy(Dev_Psi_real_previous, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaMemcpyAsync(Dev_Psi_real_previous_p, Dev_Psi_real_previous, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[1]);

   /* cudaMemcpy(Dev_Psi_2048_previous_p, Dev_Psi_2048_previous, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();*/
	
	// save function to save variables, combined so we always save in the same way and location
    // use Dev_Hat_fine for W_fine, this works because just at the end of conservation it is overwritten
	kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
			(cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
			Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
	save_variables(workspace, file_name, "0", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
    // compute conservation for first step
    compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);
    count_mesure+=1;

    cudaDeviceSynchronize();

    // repeat everything for specific defined grid
    if (use_set_grid == 1) {
//		kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//				Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//				Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//				Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
//				Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
//		cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, workspace, file_name, "vorticity_fine/w_1024_0");
//
//		//Laplacian initial
//
//		Laplacian_vort(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_2048, Dev_lap_fine_2048_real, Dev_lap_fine_2048_complex, Dev_lap_fine_2048_hat, cufftPlan_2048);
//
//		cudaMemcpy(Host_lap_fine_2048, Dev_lap_fine_2048_real, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//		cudaDeviceSynchronize();
//		writeAllRealToBinaryFile(Grid_2048.N, Host_lap_fine_2048, workspace, file_name, "vorticity_fine_lagrangian/w_lagr_0");
//
//
//		// They're everywhere ! need function
//
//		kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//				Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//				Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//				Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
//				Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
//		cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, workspace, file_name, "w_2048_0");
//
//		Psi_upsampling(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Dev_Psi_2048, cufftPlan_2048);
//
//		cudaMemcpy(Host_2048_4, Dev_Psi_2048, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(4*Grid_2048.N, Host_2048_4, workspace, file_name, "Psi_2048_0");
//
//		upsample<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiX_2048, Dev_ChiY_2048, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
//
//		cudaMemcpy(Host_2048_4, Dev_ChiX_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, workspace, file_name, "ChiX_2048_0");
//		cudaMemcpy(Host_2048_4, Dev_ChiY_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, workspace, file_name, "ChiY_2048_0");
//
//		cudaMemcpy(Host_2048_4, Dev_ChiX, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, workspace, file_name, "ChiX_0");
//		cudaMemcpy(Host_2048_4, Dev_ChiY, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
//		writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, workspace, file_name, "ChiY_0");

    }

    // now lets get the particles in
    if (SettingsMain.getParticles()) {

		// Particles initialization
		Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, Dev_particles_pos);  // project 0-1 onto 0-LX

		for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
			Rescale<<<particle_block, particle_thread>>>(Nb_particles, LX, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
			Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h);

		}
		// copy velocity values onto older values
		cudaMemcpy(Dev_particles_vel_p, Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		cudaMemcpyAsync(Dev_particles_vel_p_p, Dev_particles_vel_p, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToDevice, streams[1]);

		// safe initial position of particles
        cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToHost);
        //cudaDeviceSynchronize();
        writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, workspace, file_name, "particles/fluid/particles_pos_0");
        for(int i = 1; i < Nb_Tau_p; i+=1)
            writeAllRealToBinaryFile(2*Nb_particles, &Host_particles_pos[i * 2*Nb_particles], workspace, file_name, "particles/" + std::to_string(Tau_p[i]).substr(0, std::to_string(Tau_p[i]).find(".") + 3+ 1) + "/particles_pos_0");
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

	message = "dt = " + to_str(dt); cout<<message+"\n"; logger.push(message);
	
	#ifdef TIME_TESTING
	
		cout<<"Starting time test...\n";
		clock_t begin = clock();
		
	#endif

	// first timing save before loop - this is the initialization time
	{
		clock_t step = clock();
		double diff = double(step - begin)/CLOCKS_PER_SEC;
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
			Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, SettingsMain.getParticlesTimeIntegrationNum());
			cudaMemcpy(&Dev_particles_pos_fine_dt[(loop_ctr * Nb_fine_dt_particles * 2) % (2*prod_fine_dt_particles)], &Dev_particles_pos[2*Nb_particles*taup_fine_particles], 2*Nb_fine_dt_particles*sizeof(double), cudaMemcpyDeviceToDevice);
			// loop for all tau p
			for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){

				Particle_advect_inertia_2<<<particle_block, particle_thread>>>(Nb_particles, dt,
						&Dev_particles_pos[2*Nb_particles*index_tau_p],
						&Dev_particles_vel[2*Nb_particles*index_tau_p],
						&Dev_particles_vel_p[2*Nb_particles*index_tau_p],
						&Dev_particles_vel_p_p[2*Nb_particles*index_tau_p],
						Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p,
						Grid_psi.N, Grid_psi.NX, Grid_psi.NY, Grid_psi.h,
						Tau_p[index_tau_p], SettingsMain.getParticlesTimeIntegrationNum());

			}
		}

		// map advection
		kernel_advect_using_stream_hermite<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C1, (cufftDoubleReal*)Dev_Temp_C2,
				Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_psi.NX, Grid_psi.NY, Grid_psi.h,
				t, dt, SettingsMain.getMapEpsilon(), SettingsMain.getTimeIntegrationNum(), SettingsMain.getMapUpdateOrderNum());	// time cost
		// copy new map values onto the existing map
		cudaMemcpyAsync(Dev_ChiX, (cufftDoubleReal*)Dev_Temp_C1, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[2]);
		cudaMemcpyAsync(Dev_ChiY, (cufftDoubleReal*)Dev_Temp_C2, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[3]);

        //copy Psi to Psi_previous and Psi_previous to Psi_previous_previous
        cudaMemcpy(Dev_Psi_real_previous_p, Dev_Psi_real_previous, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(Dev_Psi_real_previous, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice, streams[1]);
		

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
		#ifdef skip_remapping  // switch to disable or enable remapping for convergence stuff
			if ( false ) {
		#else
			if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() ) {
		#endif
			if(map_stack_ctr > map_stack_length*frac_mem_cpu_to_gpu*Nb_array_RAM)
			{
				message = "Stack Saturated - Exiting"; cout<<message+"\n"; logger.push(message);
				break;
			}
			
			#ifndef TIME_TESTING
				message = "Refining Map - ctr = " + to_str(loop_ctr) + " \t map_stack_ctr = "
						+ to_str(map_stack_ctr) + " ; " + to_str(stack_length_RAM) + " ; "
						+ to_str(stack_length_Nb_array_RAM) + " \t gap = " + to_str(loop_ctr - old_ctr);
				cout<<message+"\n"; logger.push(message);
				old_ctr = loop_ctr;
			#endif
			
			//adjusting initial conditions, compute vorticity hermite
			translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack,
					Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
					Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
					Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
					Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Temp_C1, Dev_Temp_C2);
			
			
			if (map_stack_ctr%map_stack_length == 0){
				stack_length_RAM++;
				message = "stack_length_RAM = " + to_str(stack_length_RAM); cout<<message+"\n"; logger.push(message);
			}
			
			if (map_stack_ctr%(frac_mem_cpu_to_gpu*map_stack_length) == 0){
				stack_length_Nb_array_RAM++;
				message = "stack_length_Nb_array_RAM = " + to_str(stack_length_Nb_array_RAM); cout<<message+"\n"; logger.push(message);
			}
			
			//saving map stack on device/host
			//cudaMemcpy(&Dev_ChiX_stack[map_stack_ctr*4*Grid_coarse.N], Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
			//cudaMemcpy(&Dev_ChiY_stack[map_stack_ctr*4*Grid_coarse.N], Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);

            switch(stack_length_Nb_array_RAM){
                case 0:
                    cudaMemcpy(&Host_ChiX_stack_RAM_0[map_stack_ctr*4*Grid_coarse.N], Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    cudaMemcpy(&Host_ChiY_stack_RAM_0[map_stack_ctr*4*Grid_coarse.N], Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    break;
				//cout<<"pos ram 0 : "<<map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)<<endl;
                case 1:
                    cudaMemcpy(&Host_ChiX_stack_RAM_1[map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)*4*Grid_coarse.N], Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    cudaMemcpy(&Host_ChiY_stack_RAM_1[map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)*4*Grid_coarse.N], Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    break;
				//cout<<"pos ram 1 : "<<map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)<<endl;
                case 2:
                    cudaMemcpy(&Host_ChiX_stack_RAM_2[map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)*4*Grid_coarse.N], Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    cudaMemcpy(&Host_ChiY_stack_RAM_2[map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)*4*Grid_coarse.N], Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    break;

                case 3:
                    cudaMemcpy(&Host_ChiX_stack_RAM_3[map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)*4*Grid_coarse.N], Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    cudaMemcpy(&Host_ChiY_stack_RAM_3[map_stack_ctr%(frac_mem_cpu_to_gpu * map_stack_length)*4*Grid_coarse.N], Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                    break;
            }
			
			//resetting map
			kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
			
			map_stack_ctr++;
		}
		
		//loop counters
		t += dt;
		loop_ctr ++;
		//if(loop_ctr > 100)								// !!!!!!!!!!!!!!!ATTENTION!!!!!!!!!!!!!!!!!!!
			//break;
		

		#ifndef TIME_TESTING
		
			/*******************************************************************
			*						 The Final Countdown					   *
			*******************************************************************/
			/*
			if(show_progress_at < save_buffer_count)
			{
				if( loop_ctr % show_progress_at == 0 )
				{
					int p = (loop_ctr % save_buffer_count) / show_progress_at;
					int q = save_buffer_count / show_progress_at;
					double r = (double)( p * 100.0) / ( (double) q );
					
					if(r == 0)
						cout<<"100%";
					else
						cout<<r<<"%";
					cout<<endl;
					}
			}
			*/
			
			/*******************************************************************
			*							 Save snap shot						   *
			*******************************************************************/



//           if(loop_ctr == 110 || loop_ctr == 126){  // For validation of the code with the JCP paer : Spectrum at time t~=3.5, t~=4
//                kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//                		Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//						Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//						Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
//						Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, simulation_num_c);
//                cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//                writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_" + std::to_string(loop_ctr));
//            }

			if( loop_ctr % save_buffer_count == 0 )
			{
				message = "Saving Image - ctr = " + to_str(loop_ctr) + " \t save_ctr = " + to_str(save_ctr)
						+ " \t time = " + to_str(t) + " \t Compute time = " + to_str(double(clock()-begin)/CLOCKS_PER_SEC);
				cout<<message+"\n"; logger.push(message);

				// save function to save variables, combined so we always save in the same way and location
				kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
						Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
						Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
						(cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
						Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
				save_variables(workspace, file_name, to_str(save_ctr), Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
			    // compute conservation for first step
			    compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);
			    count_mesure+=1;

				// save particle positions
				if (SettingsMain.getParticles()) {
					// safe fine particles, 1 file for all positions
                    cudaMemcpy(Host_particles_pos_fine_dt, Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(double), cudaMemcpyDeviceToHost);
                    writeAllRealToBinaryFile(2*prod_fine_dt_particles, Host_particles_pos_fine_dt, workspace, file_name, "particles/fluid/particles_pos_fine_dt_" + to_str(save_ctr));
                    if (save_ctr>=1){
                        if (save_ctr%1==0){
                            cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToHost);
                            //cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, workspace, file_name, "particles/fluid/particles_pos_" + to_str(save_ctr));
                            for(int i = 1; i < Nb_Tau_p; i+=1)
                                writeAllRealToBinaryFile(2*Nb_particles, &Host_particles_pos[i * 2*Nb_particles], workspace, file_name, "particles/" + std::to_string(Tau_p[i]).substr(0, std::to_string(Tau_p[i]).find(".") + 3+ 1) + "/particles_pos_" + to_str(save_ctr));
                        }
                    }
                }

				if (save_ctr%1==0){

                    if (use_set_grid == 1) {
//						kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//								Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//								Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//								Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
//								Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
//						cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//						writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, workspace, file_name, "vorticity_fine/w_1024_" + ss.str());
                    }

//                    kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_plot, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//                    		Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//							Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//							Dev_W_plot, Dev_Complex_plot, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM,
//							Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_plot.NX, Grid_plot.NY, Grid_plot.h, Dev_W_H_initial);
//                    cudaMemcpy(Host_W_plot, Dev_W_plot, Grid_plot.sizeNReal, cudaMemcpyDeviceToHost);
//                    cudaDeviceSynchronize();
//                    writeAllRealToBinaryFile(Grid_plot.N, Host_W_plot, simulationName, "vorticity_fine/w_plot_" + ss.str());
				}
				/*
				if (save_ctr%50==0){
					for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
						cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);
				}
	            */
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
		
		#endif

		// save timing at last part of a step to take everything into account
		{
			clock_t step = clock();
			double diff = double(step - begin)/CLOCKS_PER_SEC;
			time_values[loop_ctr] = diff; // loop_ctr was already increased but first entry is init time
		}
		message = "Step : " + to_str(loop_ctr) + " ,\t Incomp Error : " + to_str(incomp_error[loop_ctr-1])
				+ " ,\t Time : " + to_str(time_values[loop_ctr]); cout<<message+"\n"; logger.push(message);
	}
	
	
	
	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/
	
	// save function to save variables, combined so we always save in the same way and location, last step so we can actually modify the condition
	kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
			(cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu,
			Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
	save_variables(workspace, file_name, "final", Host_save, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, Dev_Psi_real, Dev_ChiX, Dev_ChiY, &Grid_fine, &Grid_coarse, &Grid_psi);
	// compute conservation
	compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Host_save, Dev_Psi_real, Dev_W_coarse, (cufftDoubleReal*)Dev_Temp_C1, cufftPlan_coarse, cufftPlan_fine, Dev_Temp_C1, Dev_Temp_C2, Mesure, Mesure_fine, count_mesure);
	count_mesure+=1;

    if (SettingsMain.getParticles()) {
    	writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, workspace, file_name, "particles_pos_final");
	}

	// save all conservation data
	writeAllRealToBinaryFile(3*mes_size, Mesure, workspace, file_name, "Mesure");
	writeAllRealToBinaryFile(3*mes_size, Mesure_fine, workspace, file_name, "Mesure_fine");

    // save imcomp error
	writeAllRealToBinaryFile(iterMax, incomp_error, workspace, file_name, "Incompressibility_check");
	
	
	/*******************************************************************
	*					  Zoom on the last frame					   *
	*******************************************************************/
	
//	Zoom(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack,
//			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//			Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM,
//			Dev_W_fine, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, simulationName, LX);
	
	
	/*******************************************************************
	*						 Finalisation Nicolas					   *
	*******************************************************************/
	
	
//	kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
//			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
//			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
//			Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse.NX,
//			Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial);
//	cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "w_2048_final");
//
//	k_real_to_comp<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_W_2048, Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
//	cufftExecZ2Z(cufftPlan_2048, Dev_Complex_fine_2048, Dev_Hat_fine_bis_2048, CUFFT_FORWARD);
//	k_normalize<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Grid_2048.NX, Grid_2048.NY);
//
//	// Forming Psi hermite
//	k_fft_iLap<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// Inverse laplacian in Fourier space
//	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
//	k_comp_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Psi_2048, Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
//
//	k_fft_dy<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_2048, Dev_Hat_fine_bis_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// y-derivative of the vorticity in Fourier space
//	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_bis_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
//	k_comp_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(&Dev_Psi_2048[2*Grid_2048.N], Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
//
//	k_fft_dx<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_2048, Dev_Hat_fine_bis_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// x-derivative of the vorticity in Fourier space
//	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_bis_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
//	k_comp_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(&Dev_Psi_2048[Grid_2048.N], Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
//
//	k_fft_dy<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// y-derivative of x-derivative of of the vorticity in Fourier space
//	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
//	k_comp_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(&Dev_Psi_2048[3*Grid_2048.N], Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
//
//	cudaMemcpy(Host_2048_4, Dev_Psi_2048, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(4*Grid_2048.N, Host_2048_4, simulationName, "Psi_2048_final");
//
//	upsample<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiX_2048, Dev_ChiY_2048, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
//
//	cudaMemcpy(Host_2048_4, Dev_ChiX_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "ChiX_2048_final");
//	cudaMemcpy(Host_2048_4, Dev_ChiY_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "ChiY_2048_final");
//
//	cudaMemcpy(Host_2048_4, Dev_ChiX, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, simulationName, "ChiX_final");
//	cudaMemcpy(Host_2048_4, Dev_ChiY, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, simulationName, "ChiY_final");
	
	
	/*******************************************************************
	*						 Finalisation							   *
	*******************************************************************/
	
	
	#ifdef TIME_TESTING
	
		clock_t end = clock();
		double diff = double(end - begin)/CLOCKS_PER_SEC;
		printf("End.\nTotal time = %f\n", diff);
	
	#endif

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
	
	// Chi_stack
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
	
	// Vorticity
	cudaFree(Dev_W_coarse);
	cudaFree(Dev_W_H_fine_real);
	
	#ifdef DISCRET
		cudaFree(Dev_W_H_Initial);
	#endif

	// Psi
	cudaFree(Dev_Psi_real);
	cudaFree(Dev_Psi_real_previous);
	cudaFree(Dev_Psi_real_previous_p);
	
	cudaFree(Dev_w_min);
	cudaFree(Dev_w_max);

	// CuFFT plans
	cufftDestroy(cufftPlan_coarse);
	cufftDestroy(cufftPlan_fine);

	if (SettingsMain.getParticles()) {
	// Particles variables
	    delete [] Host_particles_pos;
	    delete [] Host_particles_vel;
	    delete [] Host_particles_pos_fine_dt;
	    cudaFree(Dev_particles_pos);
        cudaFree(Dev_particles_vel);
        cudaFree(Dev_particles_pos_fine_dt);
	}

    cudaFree(Mesure);
    cudaFree(Mesure_fine);

	// save timing at last part of a step to take everything into account
    {
		clock_t step = clock();
		double diff = double(step - begin)/CLOCKS_PER_SEC;
		time_values[loop_ctr+1] = diff;
    }
    // save timing to file
	writeAllRealToBinaryFile(iterMax+2, time_values, workspace, file_name, "Timing_Values");

	message = "Finished - Last Cuda Error = " + to_str(cudaGetLastError()) + " , Time = " + to_str(double(clock()-begin)/CLOCKS_PER_SEC); cout<<message+"\n"; logger.push(message);
}


/*******************************************************************
*							 Remapping							   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine,
		double *Dev_ChiX_stack, double *Dev_ChiY_stack,
		double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0,
		double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1,
		double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2,
		double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3,
		double *Dev_ChiX, double *Dev_ChiY,
		int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM,
		double *W_H_real, cufftHandle cufftPlan_fine, double *W_initial, int simulation_num_c,
		cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2)
{
	
	// Vorticity on coarse grid to vorticity on fine grid with a very long command, use Dev_Hat_fine for W_fine
	kernel_apply_map_stack_to_W_part_All(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY,
			Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1,
			Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3,
			(cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, stack_length, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM,
			Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h,
			W_initial, simulation_num_c);
	//kernel_apply_map_stack_to_W<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, W_real, stack_length, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
	k_real_to_comp<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>((cufftDoubleReal*)Dev_Temp_C1, Dev_Temp_C2, Grid_fine->NX, Grid_fine->NY);
	cufftExecZ2Z(cufftPlan_fine, Dev_Temp_C2, Dev_Temp_C1, CUFFT_FORWARD);
	k_normalize<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Temp_C1, Grid_fine->NX, Grid_fine->NY);

	// cutoff for turbulence spectrum
	k_fft_cut_off_scale<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Temp_C1, Grid_fine->NX, (double)(Grid_coarse->NX)/3.0);
	
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
	k_fft_cut_off_scale<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Grid_coarse->NX, (double)(Grid_psi->NX)/3.0);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C2, (cufftDoubleComplex*)Psi_real, CUFFT_INVERSE);
	k_comp_to_real<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>((cufftDoubleComplex*)Psi_real, W_real, Grid_coarse->NX, Grid_coarse->NY);

	// zero padding by moving all entries and creating a middle zone with zeros
	// initialize zeros for padding for trash variable to be used, Grid_psi is needed but it can be set for fine too
	cudaMemset(Dev_Temp_C1, 0, Grid_fine->sizeNComplex);
	k_fft_grid_add<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse->NX, Grid_psi->NX);

	// cut high frequencies for psi in fourier space, however not that much happens after zero move add from coarse grid
	k_fft_cut_off_scale<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C1, Grid_psi->NX, freq_cut_psi);

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
	k_fft_cut_off_scale<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Grid_psi->NX, (double)(Grid_psi->NX)/3.0);
	k_fft_grid_remove<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_coarse->NX, Grid_psi->NX);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Temp_C1, (cufftDoubleComplex*)Psi_real, CUFFT_INVERSE);
	k_comp_to_real<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>((cufftDoubleComplex*)Psi_real, W_real, Grid_coarse->NX, Grid_coarse->NY);

	// cut high frequencies for psi in fourier space
	k_fft_cut_off_scale<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Grid_psi->NX, freq_cut_psi);

	// Forming Psi hermite on psi grid
	k_fft_iLap<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Temp_C2, Dev_Temp_C1, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);												// Inverse laplacian in Fourier space
	fourier_hermite(Grid_psi, Dev_Temp_C1, Psi_real, Dev_Temp_C2, cufftPlan_psi);
}
// debugging lines, could be needed here to check psi
//	cudaMemcpy(Host_Debug, Psi_real, 4*Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
//	writeAllRealToBinaryFile(4*Grid_psi->N, Host_Debug, "psi_debug_4_nodes_C512_F2048_t64_T1", "Debug_2");


// sample psi on a fixed grid
void psi_upsampling(TCudaGrid2D *Grid, double *Dev_W,cufftDoubleComplex *Dev_Temp_C1, cufftDoubleComplex *Dev_Temp_C2, double *Dev_Psi, cufftHandle cufftPlan){
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
	k_comp_to_real<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>((cufftDoubleComplex*)Dev_Output, &Dev_Output[2*Grid->N], Grid->NX, Grid->NY);
	// reuse values from Trash_C2 and create dxdy, store in position 4/4
	k_fft_dx<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C2, (cufftDoubleComplex*)Dev_Output, Grid->NX, Grid->NY, Grid->h);
	cufftExecZ2Z(cufftPlan, (cufftDoubleComplex*)Dev_Output, Dev_Temp_C2, CUFFT_INVERSE);
	k_comp_to_real<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C2, &Dev_Output[3*Grid->N], Grid->NX, Grid->NY);
	// we can go back to Trash_C1 and store the other two values normally, first normal values
	cufftExecZ2Z(cufftPlan, Dev_Temp_C1, Dev_Temp_C2, CUFFT_INVERSE);
	k_comp_to_real<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C2, Dev_Output, Grid->NX, Grid->NY);
	// now compute dx derivative and store it
	k_fft_dx<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C1, Dev_Temp_C2, Grid->NX, Grid->NY, Grid->h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan, Dev_Temp_C2, Dev_Temp_C1, CUFFT_INVERSE);
	k_comp_to_real<<<Grid->blocksPerGrid, Grid->threadsPerBlock>>>(Dev_Temp_C1, &Dev_Output[Grid->N], Grid->NX, Grid->NY);
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
		// fine grid
		//Compute_Energy<<<Grid_2048.blocksPerGrid,Grid_2048.threadsPerBlock>>>(&Mesure_fine[3*count_mesure], Dev_Psi_2048, Grid_2048.N, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
		Compute_Enstrophy<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(&Mesure_fine[1 + 3*count_mesure], Dev_W_H_fine, Grid_fine->N, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
	#else
		// coarse grid
		cudaMemcpy(Host_save, Dev_Psi, 4*Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
		Compute_Energy_Host(&Mesure[3*count_mesure], Host_save, Grid_psi->N, Grid_psi->h);
		cudaMemcpy(Host_save, Dev_W_coarse, Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
		Compute_Enstrophy_Host(&Mesure[1 + 3*count_mesure], Host_save, Grid_coarse->N, Grid_coarse->h);
		// fine grid
		// missing
	#endif
	// palinstrophy is computed on Host, fine first because vorticity is saved in temporal array
	Compute_Palinstrophy_fourier(Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_H_fine, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_fine);
	Compute_Palinstrophy_fourier(Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Temp_C1, Dev_Temp_C2, cufftPlan_coarse);
}


/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

void create_directory_structure(SettingsCMM SettingsMain, string file_name, double dt, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length)
{
	if (stat("data", &st) == -1) 
	{
		cout<<"A\n";
		mkdir("data", 0700);
	}
	
	//simulationName = simulationName + "_" + currentDateTime();		// Attention !
	//simulationName = simulationName + "_currentDateTime";				

	string folderName = SettingsMain.getWorkspace() + "data/" + file_name;
	
	//creating folder
	mkdir(folderName.c_str(), 0700);
	
	string folderName1 = folderName + "/all_save_data";
	mkdir(folderName1.c_str(), 0700);
	
	string fileName = folderName + "/readme.txt";
	ofstream file(fileName.c_str(), ios::out);
	
	if(!file)
	{
		cout<<"Error writting files"<<fileName<<endl;
		exit(0);
	}
	else
	{
        file<<"Simulation name \t\t: "<<SettingsMain.getSimName()<<endl;
        switch (SettingsMain.getTimeIntegrationNum()) {
			case 0: { file<<"Time integration : Euler explicit"<<endl; break; }
			case 1: { file<<"Time integration : Adam Bashfords 2"<<endl; break; }
			case 2: { file<<"Time integration : Runge Kutta 3"<<endl; break; }
			case 3: { file<<"Time integration : Runge Kutta 4"<<endl; break; }
			default: { file<<"Time integration : Default (Euler explicit)"<<endl; break; }
		}

        file<<"N_coarse(resolution coarse grid) \t\t: "<<SettingsMain.getGridCoarse()<<endl;
		file<<"N_fine(resolution fine grid) \t\t: "<<SettingsMain.getGridFine()<<endl;
		file<<"N_psi(resolution psi grid) \t\t: "<<SettingsMain.getGridPsi()<<endl;
		file<<"time step dt \t\t: "<<dt<<endl;
		file<<"Final time \t\t: "<<SettingsMain.getFinalTime()<<endl;
		file<<"save at \t\t: "<<save_buffer_count<<endl;
		file<<"progress at \t\t: "<<show_progress_at<<endl;
		file<<"iter max \t\t: "<<iterMax<<endl;
		file<<"stack len \t\t: "<<map_stack_length<<endl;
		file<<"Incomppressibility Threshold \t: "<<SettingsMain.getIncompThreshold()<<endl;
		file<<"Map advection epsilon \t: "<<SettingsMain.getMapEpsilon()<<endl;
		file<<"Map update order \t: "<<SettingsMain.getMapUpdateOrder()<<endl;
		if (SettingsMain.getUpsampleVersion() == 0) file<<"Psi upsample version \t : Only Psi"<<endl;
		else file<<"Psi upsample version \t : Vorticity and Psi"<<endl;
		file<<"Cut Psi Frequencies at \t:"<<SettingsMain.getFreqCutPsi()<<endl;

        if (SettingsMain.getParticles()) {
        	file<<"Particles enabled"<<endl;
        	file<<"Amount of particles : "<<SettingsMain.getParticlesNum()<<endl;
            switch (SettingsMain.getParticlesTimeIntegrationNum()) {
    			case 0: { file<<"Particles Time integration : Euler explicit"<<endl; break; }
    			case 1: { file<<"Particles Time integration : Euler midpoint"<<endl; break; }
    			case 2: { file<<"Particles Time integration : Runge Kutta 3"<<endl; break; }
    			case 3: { file<<"Particles Time integration : Runge Kutta 4"<<endl; break; }
    			case -2: { file<<"Particles Time integration : Nicolas Euler midpoint"<<endl; break; }
    			case -3: { file<<"Particles Time integration : Nicolas Runge Kutta 3"<<endl; break; }
    			default: { file<<"Particles Time integration : Default (Euler explicit)"<<endl; break; }
    		}
        }
        else file<<"Particles disabled"<<endl;

		file.close();
	}
}


void save_variables(string workspace, string file_name, string i_num, double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real, double *Dev_ChiX, double *Dev_ChiY, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi) {
	// Vorticity on coarse grid : W_coarse
	cudaMemcpy(Host_save, Dev_W_coarse, Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_coarse->N, Host_save, workspace, file_name, "vorticity_coarse/w_coarse_" + i_num);
	// Vorticity on fine grid : W_fine
	cudaMemcpy(Host_save, Dev_W_fine, Grid_fine->sizeNReal, cudaMemcpyDeviceToHost);
    writeAllRealToBinaryFile(Grid_fine->N, Host_save, workspace, file_name, "vorticity_fine/w_fine_" + i_num);
	// Stream function on psi grid : Psi
	cudaMemcpy(Host_save, Dev_Psi_real, 4*Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(4*Grid_psi->N, Host_save, workspace, file_name, "stream_function/Psi_" + i_num);
	// map in x direction on coarse grid : ChiX
	cudaMemcpy(Host_save, Dev_ChiX, 4*Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(4*Grid_coarse->N, Host_save, workspace, file_name, "map_coarse/ChiX_" + i_num);
	// map in y direction on coarse grid : ChiY
	cudaMemcpy(Host_save, Dev_ChiY, 4*Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(4*Grid_coarse->N, Host_save, workspace, file_name, "map_coarse/ChiY_" + i_num);
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



























