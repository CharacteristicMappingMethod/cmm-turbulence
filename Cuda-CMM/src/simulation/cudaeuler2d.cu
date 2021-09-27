#include "cudaeuler2d.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

struct stat st = {0};



//void cuda_euler_2d(string intial_condition, int grid_scale, int fine_grid_scale, string time_integration, string map_update_order, int molly_stencil, double final_time_override, double time_step_factor)
void cuda_euler_2d(SettingsCMM SettingsMain, double final_time_override, double time_step_factor)
{
	
	/*******************************************************************
	*						 	 Constants							   *
	*******************************************************************/
	
	double LX;															// domain length
	int NX_coarse = SettingsMain.getGridCoarse();						// coarse grid size
	int NY_coarse = SettingsMain.getGridCoarse();						// coarse grid size
	int NX_fine = SettingsMain.getGridFine();							// fine grid size
	int NY_fine = SettingsMain.getGridFine();							// fine grid size
	int NX_psi = SettingsMain.getGridPsi();								// psi grid size
	int NY_psi = SettingsMain.getGridPsi();								// psi grid size
	
	double grid_by_time;
	double t0, dt, tf;													// time - initial, step, final
	int iterMax;														// maximum iteration count
	string simulationName = SettingsMain.getInitialCondition();			// name of the simulation: files get stored in the directory of this name
	int snapshots_per_second;
	int save_buffer_count;												// iterations after which files should be saved
	int map_stack_length;												// this parameter is set to avoide memory overflow on GPU
	int show_progress_at;
	
	//GPU dependent parameters
	int mem_RAM_GPU_remaps = SettingsMain.getMemRamGpuRemaps(); 		// mem_index in MB on the GPU
	int mem_RAM_CPU_remaps = SettingsMain.getMemRamCpuRemaps();			// mem_RAM_CPU_remaps in MB on the CPU
	int Nb_array_RAM = SettingsMain.getNbArrayRam();					// fixed for four different stacks
	int use_set_grid = 0;  // change later, use 2048 grid thingys or not
	
	//initialization of parameters
	LX = twoPI;	
	t0 = 0.0;

	// time steps per second used by 4_nodes
	double tmp_4nodes = 64;

	// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"
	if(simulationName == "4_nodes")
	{
		grid_by_time = 1.0;
		snapshots_per_second = 1;
		dt = 1.0/ tmp_4nodes;//(NX_coarse * grid_by_time);
		tf = 1;
	}
	else if(simulationName == "quadropole")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 20;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 50;
	}
	else if(simulationName == "two_votices")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 1;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 10;
	}
	else if(simulationName == "three_vortices")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 10;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 100;
	}
	else if(simulationName == "single_shear_layer")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 1;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 50; //50;//0.5;//100;//300;//
	}
	else if(simulationName == "turbulence_gaussienne")
	{
		grid_by_time = 1.953125;
		snapshots_per_second = 2;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 50;
	}
	else 
	{
		cout<<"Unexpected problem name specified\n";
		return;
	}
	
	//parameter overrides
	if(final_time_override > 0)
	{
		tf = final_time_override;
	}
	
	if(time_step_factor != 1)
	{
		dt *= time_step_factor;
	}
	
	
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
	save_buffer_count = tmp_4nodes/snapshots_per_second;//(NX_coarse * grid_by_time) / snapshots_per_second;//	tmp/snapshots_per_second;//(NX_coarse * grid_by_time) / snapshots_per_second;							// the denominator is snapshots per second
	show_progress_at = (32 * 4 * pow(128, 3.0)) / pow(NX_coarse, 3.0);
		if(show_progress_at < 1) show_progress_at = 1;
	map_stack_length = (mem_RAM_GPU_remaps * pow(128, 2.0))/ (double(NX_coarse * NX_coarse));
	int frac_mem_cpu_to_gpu = int(double(mem_RAM_CPU_remaps)/double(mem_RAM_GPU_remaps)/double(Nb_array_RAM));  // define how many more remappings we can save on CPU than on GPU
	
	cout<<"Initial condition : "<<simulationName<<endl;
	cout<<"Iter max : "<<iterMax<<endl;
	cout<<"Save buffer count : "<<save_buffer_count<<endl;
	cout<<"Progress at : "<<show_progress_at<<endl;
	cout<<"Map stack length : "<<map_stack_length<<endl;
	cout<<"Map stack length on RAM : "<<frac_mem_cpu_to_gpu * map_stack_length<<endl;
	cout<<"Map stack length total on RAM : "<<frac_mem_cpu_to_gpu * map_stack_length * Nb_array_RAM<<endl;
	
	// create naming for folder structure, dependent on time integration
	string sim_name_addition;
	if (SettingsMain.getTimeIntegration() == "EulerExp") {
		sim_name_addition = "_DEV_EulExp_";
	}
	else if (SettingsMain.getTimeIntegration() == "ABTwo") {
		sim_name_addition = "_DEV_AB2_";
	}
	else if (SettingsMain.getTimeIntegration() == "RKThree") {
		sim_name_addition = "_RK3_";
	}
	else if (SettingsMain.getTimeIntegration() == "RKFour") {
		sim_name_addition = "_RK4_";
	}
	else {
		sim_name_addition = "_DEV_UNNOWN_";
	}

	simulationName = simulationName + sim_name_addition + "C" + std::to_string(NX_coarse) + "_F" + std::to_string(NX_fine) + "_t" + std::to_string(tmp_4nodes).substr(0, std::to_string(tmp_4nodes).find(".")) + "_T" + std::to_string(tf).substr(0, std::to_string(tf).find(".")); //"vortex_shear_1000_4";
    simulationName = create_directory_structure(simulationName, NX_coarse, NX_fine, dt, tf, save_buffer_count, show_progress_at, iterMax, map_stack_length, SettingsMain.getIncompThreshold(), SettingsMain.getTimeIntegration());
    Logger logger(simulationName);
    cout<<"Name of simulation : "<<simulationName<<endl;
	
	
	/*******************************************************************
	*							Grids								   *
	* 	One a coarse grid where we compute derivatives and one large   *
	* 	grid where we interpolate using Hermite Basis functions.       *
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
	*							Trash variable	   					   *
	*******************************************************************/
	
	// set after largest grid, psi grid can be upsampled larger than fine grid
	long long int size_max = std::max(Grid_fine.sizeNComplex, Grid_psi.sizeNComplex);
	cufftDoubleComplex *Dev_Complex_fine, *Dev_Hat_fine, *Dev_Hat_fine_bis;
	cudaMalloc((void**)&Dev_Complex_fine, size_max);
	cudaMalloc((void**)&Dev_Hat_fine, size_max);
	cudaMalloc((void**)&Dev_Hat_fine_bis, size_max);
	
	
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
	
	printf("kernel_fft_dx\n");
	kernel_fft_dx<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
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
	
	printf("kernel_fft_dy\n");
	kernel_fft_dy<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
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
	*	Chi_new holds the new Chi values during the map advection      *
	* 																   *
	*******************************************************************/
	
	double *Host_ChiX, *Host_ChiY;
	double *Dev_ChiX, *Dev_ChiY, *Dev_Chi_new_X, *Dev_Chi_new_Y;
	
	Host_ChiX = new double[4*Grid_coarse.N];
	Host_ChiY = new double[4*Grid_coarse.N];
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_Chi_new_X, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_Chi_new_Y, 4*Grid_coarse.sizeNReal);
	
	
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
	cout<<"Map Stack Initialized"<<endl;
	logger.push("Map Stack Initialized");
	
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
	
	double *Host_W_coarse, *Host_W_fine, *Dev_W_coarse, *Dev_W_fine, *Dev_W_H_fine_real;
	
	Host_W_coarse = new double[Grid_coarse.N];
	cudaMalloc((void**)&Dev_W_coarse, Grid_coarse.sizeNReal);
	
	Host_W_fine = new double[Grid_fine.N];
	cudaMalloc((void**)&Dev_W_fine, Grid_fine.sizeNReal);
	
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
		
		kernel_real_to_complex<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_W_fine, Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, CUFFT_FORWARD);
		kernel_normalize<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Hat_fine, Grid_fine.NX, Grid_fine.NY);
		
		// Hermite vorticity array : [vorticity, x-derivative, y-derivative, xy-derivative]
		cudaMemcpy(Dev_W_H_initial, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToDevice);
		
		kernel_fft_dy<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Hat_fine, Dev_Hat_fine_bis, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);													// y-derivative of the vorticity in Fourier space
		cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine_bis, Dev_Complex_fine, CUFFT_INVERSE);
		kernel_complex_to_real  <<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Dev_W_H_initial[2*Grid_fine.N], Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		
		kernel_fft_dx<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Hat_fine, Dev_Hat_fine_bis, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);													// x-derivative of the vorticity in Fourier space
		cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine_bis, Dev_Complex_fine, CUFFT_INVERSE);
		kernel_complex_to_real  <<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Dev_W_H_initial[Grid_fine.N], Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		
		kernel_fft_dy<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_Hat_fine_bis, Dev_Hat_fine, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);													// y-derivative of x-derivative of of the vorticity in Fourier space
		cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine, Dev_Complex_fine, CUFFT_INVERSE);
		kernel_complex_to_real  <<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Dev_W_H_initial[3*Grid_fine.N], Dev_Complex_fine, Grid_fine.NX, Grid_fine.NY);
		
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
    double *Host_Psi, *Dev_Psi_real, *Dev_Psi_real_previous, *Dev_Psi_real_previous_p;

	Host_Psi = new double[4*Grid_psi.N];
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
	
	#ifdef PARTICLES

		// initialize all memory
		int Nb_Tau_p = 21;
		double Tau_p[Nb_Tau_p] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.25, 0.5, 0.75, 1, 2, 5, 13};
		int Nb_particles = 1024 * 512; //64; // *512 ;
		int particle_thread =  256;
		int particle_block = Nb_particles / particle_thread;
		printf("Nb_particles : %d\n", Nb_particles);
		double *Host_particles_pos ,*Dev_particles_pos, *Host_particles_vel ,*Dev_particles_vel;
		Host_particles_pos = new double[2*Nb_particles*Nb_Tau_p];
		Host_particles_vel = new double[2*Nb_particles*Nb_Tau_p];
		cudaMalloc((void**) &Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(double));
		cudaMalloc((void**) &Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(double));

		// create initial positions from random distribution
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p);

		// copy all starting positions onto the other tau values
		for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
			cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);

		// particles where every time step the position will be saved
		int Nb_fine_dt_particles = 1000;
		int freq_fine_dt_particles = save_buffer_count; // redundant
		int prod_fine_dt_particles = Nb_fine_dt_particles * freq_fine_dt_particles;
		double *Dev_particles_pos_fine_dt, *Host_particles_pos_fine_dt;

		Host_particles_pos_fine_dt = new double[2*prod_fine_dt_particles];
		cudaMalloc((void**) &Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(double));

	#endif


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	int count_mesure = 0;
	int mes_size = tf*snapshots_per_second + 2;  // add initial and last step
    double *Mesure;
    double *Mesure_fine;
    double incomp_error [iterMax+1];
	cudaMallocManaged(&Mesure, 3*mes_size*sizeof(double));
	cudaMallocManaged(&Mesure_fine, 3*mes_size*sizeof(double));


    // File organization : Might be moved
    std::string fi, element[5] = {"particles", "vorticity_coarse", "vorticity_fine", "Stream_function", "vorticity_fine_lagrangian"};
    for(int i = 0; i<5; i+=1){
        fi = "data/" + simulationName + "/all_save_data/" + element[i];
        mkdir(fi.c_str(), 0700);
    }

    #ifdef PARTICLES
        fi = "data/" + simulationName + "/all_save_data/particles/fluid";
        mkdir(fi.c_str(), 0700);
        for(int i = 1; i<Nb_Tau_p; i+=1){
            fi = "data/" + simulationName + "/all_save_data/particles/" + std::to_string(Tau_p[i]).substr(0, std::to_string(Tau_p[i]).find(".") + 3+ 1);
            mkdir(fi.c_str(), 0700);
        }
    #endif


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
	*						  Last Cuda Error						   *
	*******************************************************************/
	
	cout<<cudaGetErrorName (cudaGetLastError());						// cudaErrorMemoryAllocation
	printf("\n");

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
	translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Dev_W_fine, Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
	
	//in step 1, Psi_p stays the same old one, so we compute psi0 in psi_p instead
	// compute psi and store it in psi_p
    evaluate_stream_hermite(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real_previous, cufftPlan_coarse, cufftPlan_psi, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, SettingsMain.getMollyStencil());
    //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
    // set psi_p_p as psi_p
    cudaMemcpy(Dev_Psi_real_previous_p, Dev_Psi_real_previous, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
   /* cudaMemcpy(Dev_Psi_2048_previous_p, Dev_Psi_2048_previous, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();*/
	
	//copying data on host to save initial stream function
    cudaMemcpy(Host_Psi, Dev_Psi_real_previous, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(4*Grid_psi.N, Host_Psi, simulationName, "Stream_function/Psi_0");

	// copy vorticity on host to save initial vorticity
	cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "vorticity_fine/w_fine_0");
	cudaMemcpy(Host_W_coarse, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);	
	writeAllRealToBinaryFile(Grid_coarse.N, Host_W_coarse, simulationName, "vorticity_coarse/w_coarse_0");
    cudaDeviceSynchronize();

    // repeat everything for specific defined grid
    if (use_set_grid == 1) {
		kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
		cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_0");

		//Laplacian initial

		Laplacian_vort(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_2048, Dev_lap_fine_2048_real, Dev_lap_fine_2048_complex, Dev_lap_fine_2048_hat, cufftPlan_2048);

		cudaMemcpy(Host_lap_fine_2048, Dev_lap_fine_2048_real, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		writeAllRealToBinaryFile(Grid_2048.N, Host_lap_fine_2048, simulationName, "vorticity_fine_lagrangian/w_lagr_0");


		// They're everywhere ! need function

		kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
		cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "w_2048_0");

		Psi_upsampling(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Dev_Psi_2048, cufftPlan_2048);

		cudaMemcpy(Host_2048_4, Dev_Psi_2048, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_2048.N, Host_2048_4, simulationName, "Psi_2048_0");

		upsample<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiX_2048, Dev_ChiY_2048, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);

		cudaMemcpy(Host_2048_4, Dev_ChiX_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "ChiX_2048_0");
		cudaMemcpy(Host_2048_4, Dev_ChiY_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "ChiY_2048_0");

		cudaMemcpy(Host_2048_4, Dev_ChiX, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, simulationName, "ChiX_0");
		cudaMemcpy(Host_2048_4, Dev_ChiY, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, simulationName, "ChiY_0");

    }

    // compute conservation for first step
    compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Dev_Psi_real_previous, Host_Psi, Dev_W_coarse, Host_W_coarse, Dev_W_fine, Host_W_fine, cufftPlan_coarse, cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, Mesure, Mesure_fine, count_mesure);
    count_mesure+=1;

    // now lets get the particles in
	#ifdef PARTICLES

		// Particles initialization
		Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, Dev_particles_pos);
		Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

		#pragma unroll Nb_Tau_p - 1
		for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
			Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
			Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
			#ifndef RKThree_PARTICLES
				Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
				// Deux fois Dev_psi_real_previous car Dev_Psi_real n'est pas encore initialisé                                                                                                                ^^           ^^

			#else
				Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
			#endif
			}
	#endif




	/////////////////////// slight different from regular loop
	//saving max and min for plotting purpose
	get_max_min(&Grid_fine, Host_W_fine, &w_min, &w_max);
	cout<<"W min = "<<w_min<<endl<<"W max = "<<w_max<<endl;
	
	double t = t0;
	int loop_ctr = 0;
	int save_ctr = 1;

	int old_ctr = 0;

	cout<<std::setprecision(30)<<"dt = "<<dt<<endl;
	
	#ifdef TIME_TESTING
	
		cout<<"Starting time test...\n";
		clock_t begin = clock();
		
	#endif

	cout<<cudaGetErrorName (cudaGetLastError());
	printf("\n");

	clock_t begin = clock();

	/*******************************************************************
	*							 Main loop							   *
	*******************************************************************/

	while(tf - t > 1e-10 && loop_ctr < iterMax)
	{
		//avoiding over-stepping for last time-step
		if(t + dt > tf)
			dt = tf - t;

		// compute stream hermite from vorticity
	    evaluate_stream_hermite(&Grid_coarse, &Grid_fine, &Grid_psi, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, cufftPlan_psi, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, SettingsMain.getMollyStencil());
        //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);


		// Particles advection
		#ifdef PARTICLES
			Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
			cudaMemcpy(&Dev_particles_pos_fine_dt[(loop_ctr * Nb_fine_dt_particles * 2) % (2*prod_fine_dt_particles)], Dev_particles_pos, 2*Nb_fine_dt_particles*sizeof(double), cudaMemcpyDeviceToDevice);
			#pragma unroll Nb_Tau_p - 1
			for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
				#ifndef RKThree_PARTICLES
					 Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
				#else
					 Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
				#endif

			}
		#endif

		// map advection
		kernel_advect_using_stream_hermite<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_Chi_new_X, Dev_Chi_new_Y, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_psi.NX, Grid_psi.NY, Grid_psi.h, t, dt, SettingsMain.getMapEpsilon(), SettingsMain.getTimeIntegrationNum(), SettingsMain.getMapUpdateOrderNum());	// time cost
		// copy new map values onto the existing map
		cudaMemcpy(Dev_ChiX, Dev_Chi_new_X, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
		cudaMemcpy(Dev_ChiY, Dev_Chi_new_Y, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);

       /* #ifdef RKThree_PARTICLES
            // copy particles velocity to previous instance
           cudaMemcpy(Dev_particles_vel_previous_p, Dev_particles_vel_previous, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(Dev_particles_vel_previous, Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToDevice);
        #endif*/

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
			kernel_incompressibility_check<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, (cufftDoubleReal*)Dev_Complex_fine, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);								// time cost		A optimiser
			// We don't need to have Dev_gradChi in memory we juste need to know if it exist a value such as : abs(this_value - 1) > inCompThreshold
			
			//cudaDeviceSynchronize();
			// compute minimum for actual check on dev, coppy to machine to get minimum from all blocks
			Dev_get_max_min<<<grad_block, grad_thread>>>(Grid_fine.N, (cufftDoubleReal*)Dev_Complex_fine, Dev_w_min, Dev_w_max);// Dev_gradChi cufftDoubleComplex cufftDoubleReal
			//cudaMemcpy(Host_w_min, Dev_w_min, sizeof(double)*grad_block*grad_thread, cudaMemcpyDeviceToHost);
			//cudaMemcpy(Host_w_max, Dev_w_max, sizeof(double)*grad_block*grad_thread, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(Host_w_min, Dev_w_min, sizeof(double)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[0]);
			cudaMemcpyAsync(Host_w_max, Dev_w_max, sizeof(double)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[1]);
			// now compute minimum in Host
			Host_get_max_min(grad_block*grad_thread, Host_w_min, Host_w_max, &grad_chi_min, &grad_chi_max);

		}
			
		//resetting map and adding to stack
		incomp_error[loop_ctr] = fmax(fabs(grad_chi_min - 1), fabs(grad_chi_max - 1));
		cout<<"Step : "<<loop_ctr<<" , Incomp Error : "<<incomp_error[loop_ctr]<<endl;
		#ifdef skip_remapping  // switch to disable or enable remapping for convergence stuff
			if ( false ) {
		#else
			if( incomp_error[loop_ctr] > SettingsMain.getIncompThreshold() ) {
		#endif
			if(map_stack_ctr > map_stack_length*frac_mem_cpu_to_gpu*Nb_array_RAM)
			{
				cout<<"Stack Saturated... Exiting .. \n";
				break;
			}
			
			#ifndef TIME_TESTING
				printf("Refining Map... ctr = %d \t map_stack_ctr = %d ; %d ; %d \t gap = %d \t incomp_err = %e\n", loop_ctr, map_stack_ctr, stack_length_RAM, stack_length_Nb_array_RAM, loop_ctr - old_ctr, incomp_error[loop_ctr]);
				snprintf(logger.buffer, sizeof(logger.buffer), "Refining Map... ctr = %d \t map_stack_ctr = %d \t gap = %d \t incomp_err = %e", loop_ctr, map_stack_ctr, loop_ctr - old_ctr, incomp_error[loop_ctr]);
				logger.push();
				old_ctr = loop_ctr;
			#endif
			
			//adjusting initial conditions
			translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Dev_W_fine, Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, SettingsMain.getInitialConditionNum(), Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
			
			
			if (map_stack_ctr%map_stack_length == 0){
				stack_length_RAM++;
				cout<<"stack_length_RAM : "<<stack_length_RAM<<endl;
			}
			
			if (map_stack_ctr%(frac_mem_cpu_to_gpu*map_stack_length) == 0){
				stack_length_Nb_array_RAM++;
				cout<<"stack_length_Nb_array_RAM : "<<stack_length_Nb_array_RAM<<endl;
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
//                kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, simulation_num_c);
//                cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
//                writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_" + std::to_string(loop_ctr));
//
//            }

			if( loop_ctr % save_buffer_count == 0 )
			{
			  printf("Saving Image... ctr = %d \t save_ctr = %d  \t time = %f \t  Compute time : %lf \n", loop_ctr, save_ctr, t, double(clock()-begin)/CLOCKS_PER_SEC);
				snprintf(logger.buffer, sizeof(logger.buffer), "Saving Image... ctr = %d \t save_ctr = %d  \t time = %f", loop_ctr, save_ctr, t); 
				logger.push();
				
				//copying data on host
				cudaMemcpy(Host_W_coarse, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
				cudaMemcpy(Host_Psi, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                #ifdef PARTICLES
                    cudaMemcpy(Host_particles_pos_fine_dt, Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(double), cudaMemcpyDeviceToHost);
                #endif
                //writing to file
				std::ostringstream ss;
				ss<<save_ctr;

				writeAllRealToBinaryFile(Grid_coarse.N, Host_W_coarse, simulationName, "vorticity_coarse/w_coarse_" + ss.str());
				writeAllRealToBinaryFile(4*Grid_coarse.N, Host_Psi, simulationName, "Stream_function/Psi_" + ss.str());
                #ifdef PARTICLES

                    writeAllRealToBinaryFile(2*prod_fine_dt_particles, Host_particles_pos_fine_dt, simulationName, "particles/fluid/particles_pos_fine_dt_" + ss.str());
                    if (save_ctr>=1){
                        if (save_ctr%1==0){
                            cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToHost);
                            //cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles/fluid/particles_pos_" + ss.str());
                            /*cudaMemcpy(Host_particles_pos, Dev_particles_pos_1, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles/fluid/particles_pos_dt_" + ss.str());
                            cudaMemcpy(Host_particles_pos, Dev_particles_pos_2, 2*Nb_particles*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles/fluid/particles_pos_dt2_" + ss.str());
			    */
                            for(int i = 1; i < Nb_Tau_p; i+=1)
                                writeAllRealToBinaryFile(2*Nb_particles, &Host_particles_pos[i * 2*Nb_particles], simulationName, "particles/" + std::to_string(Tau_p[i]).substr(0, std::to_string(Tau_p[i]).find(".") + 3+ 1) + "/particles_pos_" + ss.str());
                        }
                    }

                #endif
				if (save_ctr%1==0){
					kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_fine, Dev_Complex_fine, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
                    cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
                    writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "vorticity_fine/w_fine_" + ss.str());

                    if (use_set_grid == 1) {
						kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
						cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
						writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_" + ss.str());
                    }

                   /* kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_plot, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_plot, Dev_Complex_plot, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_plot.NX, Grid_plot.NY, Grid_plot.h, Dev_W_H_initial);

                    cudaMemcpy(Host_W_plot, Dev_W_plot, Grid_plot.sizeNReal, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    writeAllRealToBinaryFile(Grid_plot.N, Host_W_plot, simulationName, "vorticity_fine/w_plot_" + ss.str());*/
				}
				/*
				if (save_ctr%50==0){
					for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
						cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(double), cudaMemcpyDeviceToDevice);
				}
	            */
				save_ctr++;

			    // compute conservation
			    compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Dev_Psi_real_previous, Host_Psi, Dev_W_coarse, Host_W_coarse, Dev_W_fine, Host_W_fine, cufftPlan_coarse, cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, Mesure, Mesure_fine, count_mesure);
			    count_mesure+=1;

                //Laplacian_vort(&Grid_fine, Dev_W_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_lap_fine_real, Dev_lap_fine_complex, Dev_lap_fine_hat, cufftPlan_fine);

                //cudaMemcpy(Host_lap_fine, Dev_lap_fine_real, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
                //cudaDeviceSynchronize();
                //writeAllRealToBinaryFile(Grid_fine.N, Host_lap_fine, simulationName, "vorticity_fine_lagrangian/w_lagr_" + ss.str());


                 //Laplacian initial

				if (use_set_grid == 1) {
					Laplacian_vort(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_2048, Dev_lap_fine_2048_real, Dev_lap_fine_2048_complex, Dev_lap_fine_2048_hat, cufftPlan_2048);
					cudaMemcpy(Host_lap_fine_2048, Dev_lap_fine_2048_real, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
					cudaDeviceSynchronize();
					writeAllRealToBinaryFile(Grid_2048.N, Host_lap_fine_2048, simulationName, "vorticity_fine_lagrangian/w_lagr_" + ss.str());
				}

			}
			
			int error = cudaGetLastError();
			//cudaError_t err = cudaGetLastError();
			//if (err != cudaSuccess){
			//	  printf("%s\n", cudaGetErrorString(err));
			//}
			if(error != 0)
			{
				cout<<"Finished; Last Cuda Error : "<<error<<endl;
				string temp = "Finished; Last Cuda Error : " + error; 
				logger.push(temp);
				exit(0);
				break;
			}
		
		#endif
		
	}
	
	
	
	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/
	
	kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_fine, Dev_Complex_fine, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial, SettingsMain.getInitialConditionNum());
	cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "w_fine_final");
	writeAllRealToBinaryFile(4*Grid_coarse.N, Host_Psi, simulationName, "Psi_final");
    #ifdef PARTICLES
        writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles_pos_final");
    #endif

	// compute conservation
	compute_conservation_targets(&Grid_fine, &Grid_coarse, &Grid_psi, Dev_Psi_real_previous, Host_Psi, Dev_W_coarse, Host_W_coarse, Dev_W_fine, Host_W_fine, cufftPlan_coarse, cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, Mesure, Mesure_fine, count_mesure);
	count_mesure+=1;

	// save all conservation data
	writeAllRealToBinaryFile(3*mes_size, Mesure, simulationName, "Mesure");
	writeAllRealToBinaryFile(3*mes_size, Mesure_fine, simulationName, "Mesure_fine");

    // save imcomp error
	writeAllRealToBinaryFile(iterMax, incomp_error, simulationName, "Incompressibility_check");
	
	
	/*******************************************************************
	*					  Zoom on the last frame					   *
	*******************************************************************/
	
	//Zoom(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Dev_W_fine, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, simulationName, LX);
	
	
	/*******************************************************************
	*						 Finalisation Nicolas					   *
	*******************************************************************/
	
	
/*	kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial);
	cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "w_2048_final");
	
	kernel_real_to_complex<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_W_2048, Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
	cufftExecZ2Z(cufftPlan_2048, Dev_Complex_fine_2048, Dev_Hat_fine_bis_2048, CUFFT_FORWARD);	
	kernel_normalize<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Grid_2048.NX, Grid_2048.NY);
	
	// Forming Psi hermite
	kernel_fft_iLap<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// Inverse laplacian in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Psi_2048, Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
	
	kernel_fft_dy<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_2048, Dev_Hat_fine_bis_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_bis_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(&Dev_Psi_2048[2*Grid_2048.N], Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
	
	kernel_fft_dx<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_2048, Dev_Hat_fine_bis_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_bis_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(&Dev_Psi_2048[Grid_2048.N], Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);
	
	kernel_fft_dy<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_Hat_fine_bis_2048, Dev_Hat_fine_2048, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);													// y-derivative of x-derivative of of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_2048, Dev_Hat_fine_2048, Dev_Complex_fine_2048, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(&Dev_Psi_2048[3*Grid_2048.N], Dev_Complex_fine_2048, Grid_2048.NX, Grid_2048.NY);

	cudaMemcpy(Host_2048_4, Dev_Psi_2048, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(4*Grid_2048.N, Host_2048_4, simulationName, "Psi_2048_final"); 
	
	upsample<<<Grid_2048.blocksPerGrid, Grid_2048.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiX_2048, Dev_ChiY_2048, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
	
	cudaMemcpy(Host_2048_4, Dev_ChiX_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "ChiX_2048_final");
	cudaMemcpy(Host_2048_4, Dev_ChiY_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "ChiY_2048_final");
	
	cudaMemcpy(Host_2048_4, Dev_ChiX, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, simulationName, "ChiX_final");
	cudaMemcpy(Host_2048_4, Dev_ChiY, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_coarse.N, Host_2048_4, simulationName, "ChiY_final");
	
	*/
	
	
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
	
	char buffer[50];
	int cudaError = cudaGetLastError();
	ofstream errorLogFile("data/errorLog.csv", ios::out | ios::app);
	sprintf(buffer, "%e", maxError);
	errorLogFile<<NX_coarse<<", "<<fabs(dt)<<", "<<tf<<", "<<buffer<<","<<cudaError<<endl;
	errorLogFile.close();
	
	
	
	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/
	
	cudaFree(Dev_W_H_initial);
	
	// Trash variable
	cudaFree(Dev_Complex_fine);
	cudaFree(Dev_Hat_fine);
	cudaFree(Dev_Hat_fine_bis);
	
	// Chi
	delete [] Host_ChiX;
	delete [] Host_ChiY;
	cudaFree(Dev_ChiX);
	cudaFree(Dev_ChiY);
	cudaFree(Dev_Chi_new_X);
	cudaFree(Dev_Chi_new_Y);
	
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
	delete [] Host_W_coarse;
	delete [] Host_W_fine;
	cudaFree(Dev_W_coarse);
	cudaFree(Dev_W_fine);
	cudaFree(Dev_W_H_fine_real);
	
	#ifdef DISCRET
		cudaFree(Dev_W_H_Initial);
	#endif

	// Psi
	delete [] Host_Psi;
	cudaFree(Dev_Psi_real);
	cudaFree(Dev_Psi_real_previous);
	cudaFree(Dev_Psi_real_previous_p);
	
	cudaFree(Dev_w_min);
	cudaFree(Dev_w_max);

	// CuFFT plans
	cufftDestroy(cufftPlan_coarse);
	cufftDestroy(cufftPlan_fine);

    #ifdef PARTICLES
	// Particles
	    delete [] Host_particles_pos;
	    delete [] Host_particles_vel;
	    delete [] Host_particles_pos_fine_dt;
	    cudaFree(Dev_particles_pos);
        cudaFree(Dev_particles_vel);
        cudaFree(Dev_particles_pos_fine_dt);
	#endif

    cudaFree(Mesure);
    cudaFree(Mesure_fine);

	cout<<"Finished; Last Cuda Error : "<<cudaError<<endl;
}


/*******************************************************************
*							 Remapping							   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, double *Dev_ChiX_stack, double *Dev_ChiY_stack, double *Host_ChiX_stack_RAM_0, double *Host_ChiY_stack_RAM_0, double *Host_ChiX_stack_RAM_1, double *Host_ChiY_stack_RAM_1, double *Host_ChiX_stack_RAM_2, double *Host_ChiY_stack_RAM_2, double *Host_ChiX_stack_RAM_3, double *Host_ChiY_stack_RAM_3, double *Dev_ChiX, double *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, double *W_real, double *W_H_real, cufftHandle cufftPlan_fine, double *W_initial, int simulation_num_c, cufftDoubleComplex *Dev_Complex_fine, cufftDoubleComplex *Dev_Hat_fine, cufftDoubleComplex *Dev_Hat_fine_bis)
{
	
	// Vorticity on coarse grid to vorticity on fine grid
	kernel_apply_map_stack_to_W_part_All(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, W_real, Dev_Complex_fine, stack_length, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial, simulation_num_c);
	//kernel_apply_map_stack_to_W<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, W_real, stack_length, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
	kernel_real_to_complex<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, CUFFT_FORWARD);
	kernel_normalize<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Grid_fine->NX, Grid_fine->NY);
	/*
	cut_off_scale<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Grid_fine->NX);
	Dev_Hat_fine[0].x = 0;
	Dev_Hat_fine[0].y = 0;
	cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine, Dev_Complex_fine, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	*/
	cut_off_scale<<<1, 1>>>(Dev_Hat_fine, Grid_fine->NX);
	cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine, Dev_Complex_fine, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	
	// Hermite vorticity array : [vorticity, x-derivative, y-derivative, xy-derivative]
	cudaMemcpy(W_H_real, W_real, Grid_fine->sizeNReal, cudaMemcpyDeviceToDevice);
	
	kernel_fft_dy<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Dev_Hat_fine_bis, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine_bis, Dev_Complex_fine, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(&W_H_real[2*Grid_fine->N], Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	
	kernel_fft_dx<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Dev_Hat_fine_bis, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine_bis, Dev_Complex_fine, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(&W_H_real[Grid_fine->N], Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	
	kernel_fft_dy<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine_bis, Dev_Hat_fine, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);													// y-derivative of x-derivative of of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_fine, Dev_Hat_fine, Dev_Complex_fine, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(&W_H_real[3*Grid_fine->N], Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	
}


/*******************************************************************
*						 Computation of Psi						   *
*******************************************************************/

void evaluate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_psi, double *Dev_ChiX, double *Dev_ChiY, double *Dev_W_H_fine_real, double *W_real, double *Psi_real, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_psi, cufftDoubleComplex *Dev_Complex_coarse, cufftDoubleComplex *Dev_Hat_coarse, cufftDoubleComplex *Dev_Hat_coarse_bis, int molly_stencil)
{

	// apply map to w and sample using mollifier
	kernel_apply_map_and_sample_from_hermite<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, W_real, Dev_W_H_fine_real, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, molly_stencil);

	// forward fft
	kernel_real_to_complex<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Complex_coarse, Dev_Hat_coarse_bis, CUFFT_FORWARD);
	kernel_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Grid_coarse->NX, Grid_coarse->NY);

	// cut_off and W
	cut_off_scale<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Grid_coarse->NX);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_coarse_bis, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);

	// zero padding by moving all entries and creating a middle zone with zeros
	// initialize zeros for padding for trash variable to be used, Grid_psi is needed but it can be set for fine too
	cudaMemset(Dev_Hat_coarse, 0, Grid_fine->sizeNComplex);
	zero_move<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Dev_Hat_coarse, Grid_coarse->NX, Grid_psi->NX);

	// Forming Psi hermite, everything now on psi grid instead of coarse
	kernel_fft_iLap<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Hat_coarse, Dev_Hat_coarse_bis, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);												// Inverse laplacian in Fourier space
	cufftExecZ2Z(cufftPlan_psi, Dev_Hat_coarse_bis, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Psi_real, Dev_Complex_coarse, Grid_psi->NX, Grid_psi->NY);

	kernel_fft_dy<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Hat_coarse_bis, Dev_Hat_coarse, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_psi, Dev_Hat_coarse, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(&Psi_real[2*Grid_psi->N], Dev_Complex_coarse, Grid_psi->NX, Grid_psi->NY);

	kernel_fft_dx<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Hat_coarse_bis, Dev_Hat_coarse, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_psi, Dev_Hat_coarse, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(&Psi_real[Grid_psi->N], Dev_Complex_coarse, Grid_psi->NX, Grid_psi->NY);

	kernel_fft_dy<<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(Dev_Hat_coarse, Dev_Hat_coarse_bis, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);													// y-derivative of x-derivative of of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_psi, Dev_Hat_coarse_bis, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_psi->blocksPerGrid, Grid_psi->threadsPerBlock>>>(&Psi_real[3*Grid_psi->N], Dev_Complex_coarse, Grid_psi->NX, Grid_psi->NY);

}


/*******************************************************************
*		 Computation of Global conservation values				   *
*******************************************************************/

void compute_conservation_targets(TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi, double *Dev_Psi, double *Host_Psi, double *Dev_W_coarse, double *Host_W_coarse, double *Dev_W_fine, double *Host_W_fine, cufftHandle cufftPlan_coarse, cufftHandle cufftPlan_fine, cufftDoubleComplex *Dev_Complex, cufftDoubleComplex *Dev_Hat, cufftDoubleComplex *Dev_Hat_bis, double *Mesure, double *Mesure_fine, int count_mesure) {
	#ifndef sm_50
		// coarse grid
		Compute_Energy<<<Grid_psi->blocksPerGrid,Grid_psi->threadsPerBlock>>>(&Mesure[3*count_mesure], Dev_Psi, Grid_psi->N, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);
		Compute_Enstrophy<<<Grid_coarse->blocksPerGrid,Grid_coarse->threadsPerBlock>>>(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse->N, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);
		// fine grid
		//Compute_Energy<<<Grid_2048.blocksPerGrid,Grid_2048.threadsPerBlock>>>(&Mesure_fine[3*count_mesure], Dev_Psi_2048, Grid_2048.N, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
		Compute_Enstrophy<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(&Mesure_fine[1 + 3*count_mesure], Dev_W_fine, Grid_fine->N, Grid_fine->NX, Grid_fine->NY, Grid_fine->h);
	#else
		// coarse grid
		cudaMemcpy(Host_Psi, Dev_Psi, Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
		cudaMemcpy(Host_W_coarse, Dev_W_coarse, Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
		Compute_Energy_Host(&Mesure[3*count_mesure], Host_Psi, Grid_psi->N, Grid_psi->NX, Grid_psi->NY, Grid_psi->h);
		Compute_Enstrophy_Host(&Mesure[1 + 3*count_mesure], Host_W_coarse, Grid_coarse->N, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);
		// fine grid
		// missing
	#endif
	// palinstrophy is computed on Host
	Compute_Palinstrophy(Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Complex, Dev_Hat, Dev_Hat_bis, cufftPlan_coarse);
	Compute_Palinstrophy(Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_fine, Dev_Complex, Dev_Hat, Dev_Hat_bis, cufftPlan_fine);
}


/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

string create_directory_structure(string simulationName, int NX_coarse, int NX_fine, double dt, double T, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length, double inCompThreshold, string time_integration)
{
	if (stat("data", &st) == -1) 
	{
		cout<<"A\n";
		mkdir("data", 0700);
	}
	
	//simulationName = simulationName + "_" + currentDateTime();		// Attention !
	//simulationName = simulationName + "_currentDateTime";				

	string folderName = "data/" + simulationName;
	
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
        file<<"Simulation name \t\t:"<<simulationName<<endl;
        if (time_integration == "EulExp") {
        	file<<"Time integration : Euler explicit"<<endl;
        }
        else if (time_integration == "ABTwo") {
        	file<<"Time integration : Adam Bashfords 2"<<endl;
        }
        else if (time_integration == "RKThree") {
        	file<<"Time integration : Runge Kutta 3"<<endl;
        }
        else if (time_integration == "RKFour") {
        	file<<"Time integration : Runge Kutta 4"<<endl;
        }
        else file<<"Time integration : Euler explicit"<<endl;

        #ifdef PARTICLES
              file<<"Particles enabled"<<endl;
        #else
              file<<"Particles disabled"<<endl;
        #endif
        file<<"NX_coarse(resolution coarse grid) \t\t:"<<NX_coarse<<endl;
		file<<"NX_fine(resolution fine grid) \t\t:"<<NX_fine<<endl;
		file<<"time step dt \t\t:"<<dt<<endl;
		file<<"T_final \t\t:"<<T<<endl;
		file<<"save at \t:"<<save_buffer_count<<endl;
		file<<"progress at \t:"<<show_progress_at<<endl;
		file<<"iter max \t:"<<iterMax<<endl;
		file<<"stack len \t:"<<map_stack_length<<endl;
		file<<"inCompThreshold :"<<inCompThreshold<<endl;
		file.close();
	}
	
	return simulationName;
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



























