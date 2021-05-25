#include "cudaeuler2d.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

struct stat st = {0};



void cuda_euler_2d(string problem_name, int grid_scale, int fine_grid_scale, double final_time_override, double time_step_factor)
{
	
	/*******************************************************************
	*						 	 Constants							   *
	*******************************************************************/
	
	ptype LX;															// domain length
	int NX_coarse, NY_coarse;											// coarse grid size
	int NX_fine, NY_fine;												// fine grid size
	
	double grid_by_time;
	ptype t0, dt, tf;													// time - initial, step, final
	int iterMax;														// maximum iteration count 
	string simulationName;												// name of the simulation: files get stored in the directory of this name
	int snapshots_per_second;
	int save_buffer_count;												// iteractions after which files should be saved
	int map_stack_length;												// this parameter is set to avoide memory overflow on GPU
	int show_progress_at;
	ptype inCompThreshold = 0.01;										// the maximum allowance of map to deviate from grad_chi begin 1
	
	//GPU dependent parameters
	int mem_index = 1024; 												// mem_index = 1024 for 2GB GPU ram												// Need to be change in the future for large data
	
	//initialization of parameters
	simulationName = problem_name;
	LX = twoPI;	
	NX_coarse = NY_coarse = grid_scale;
	NX_fine = NY_fine = fine_grid_scale;
	t0 = 0.0;
	
	
	// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"

	if(simulationName == "4_nodes")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 2;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 1;		
	}
	else if(simulationName == "quadropole")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 20;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 50;		
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
		tf = 100; //50; //0.5;//100;//300;//
	}
	else if(simulationName == "two_votices")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 20;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 80;		
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
		snapshots_per_second = 10;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 10;	
	#endif
	#ifndef DISCRET
		ptype *Dev_W_H_initial;
		cudaMalloc((void**)&Dev_W_H_initial, 8);
	#endif
	
	
	
	//shared parameters
	iterMax = ceil(tf / dt);
	save_buffer_count = (NX_coarse * grid_by_time) / snapshots_per_second;							// the denominator is snapshots per second
	show_progress_at = (32 * 4 * pow(128.0, 3.0)) / pow(NX_coarse, 3.0); 
		if(show_progress_at < 1) show_progress_at = 1;
	map_stack_length = (mem_index * pow(128.0, 2.0))/ (double(NX_coarse * NX_coarse)); 
	
	cout<<"Simulation name : "<<simulationName<<endl;
	cout<<"Iter max : "<<iterMax<<endl;
	cout<<"Save buffer cout : "<<save_buffer_count<<endl;
	cout<<"Progress at : "<<show_progress_at<<endl;
	cout<<"map stack length : "<<map_stack_length<<endl;
	
	std::ostringstream ssN;
	ssN<< "_" << NX_coarse;
	//simulationName = simulationName + ssN.str();													// File name
	//simulationName = simulationName;									
	simulationName = "vortex_shear_1000_4";
	simulationName = create_directory_structure(simulationName, NX_coarse, NX_fine, dt, tf, save_buffer_count, show_progress_at, iterMax, map_stack_length, inCompThreshold);
	Logger logger(simulationName);
	
	
	/*******************************************************************
	*							Grids								   *
	* 	One a coarse grid where we compute derivatives and one large   *
	* 	grid where we interpolate using Hermite Basis functions.       *
	*																   *
	*******************************************************************/
	
	TCudaGrid2D Grid_coarse(NX_coarse, NY_coarse, LX);
	TCudaGrid2D Grid_fine(NX_fine, NY_fine, LX);
	
	
	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufftPlan_coarse, cufftPlan_fine;
	cufftPlan2d(&cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z);
	cufftPlan2d(&cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z);
	
	
	/*******************************************************************
	*							Complex variable					   *
	*******************************************************************/
	
	cuPtype *Dev_Complex_coarse, *Dev_Complex_fine, *Dev_Hat_coarse, *Dev_Hat_fine, *Dev_Hat_coarse_bis, *Dev_Hat_fine_bis;
	cudaMalloc((void**)&Dev_Complex_coarse, Grid_coarse.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_coarse, Grid_coarse.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_coarse_bis, Grid_coarse.sizeNComplex);
	cudaMalloc((void**)&Dev_Complex_fine, Grid_fine.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_fine, Grid_fine.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_fine_bis, Grid_fine.sizeNComplex);
	
	
	/*******************************************************************
	*							  Chi								   *
	* 	Chi is an array that contains Chi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative   					       *
	*	ChiDual is an array that contains values of Chi NE, SE, SW, NW *
	* 																   *
	*******************************************************************/
	
	ptype *Host_ChiX, *Host_ChiY;
	ptype *Dev_ChiX, *Dev_ChiY, *Dev_ChiDualX, *Dev_ChiDualY;
	
	Host_ChiX = new ptype[4*Grid_coarse.N];
	Host_ChiY = new ptype[4*Grid_coarse.N];							
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiDualX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiDualY, 4*Grid_coarse.sizeNReal);
	
	
	/*******************************************************************
	*					       Chi_stack							   *
	* 	We need to save the variable Chi to be able to make	the        *
	* 	remapping or the zoom				   					       *
	* 																   *
	*******************************************************************/
	
	ptype *Host_ChiX_stack, *Host_ChiY_stack, *Dev_ChiX_stack, *Dev_ChiY_stack;
	
	Host_ChiX_stack = new ptype[map_stack_length * 4*Grid_coarse.sizeNReal];		
	Host_ChiY_stack = new ptype[map_stack_length * 4*Grid_coarse.sizeNReal];										// Do we need to start the allocation with the largest variable ?
	cudaMalloc((void **) &Dev_ChiX_stack, map_stack_length * 4*Grid_coarse.sizeNReal);								// map_stack_length * 4 * Grid_coarse.sizeNReal is to big in the GPU memory, we need to put this in the RAM and copy part by part in GPU memory
	cudaMalloc((void **) &Dev_ChiY_stack, map_stack_length * 4*Grid_coarse.sizeNReal);
	int map_stack_ctr = 0;
	cout<<"Map Stack Initialized"<<endl;
	logger.push("Map Stack Initialized");
	
	
	/*******************************************************************
	*					       Vorticity							   *
	* 	We need to have different variable version. coarse/fine,       *
	* 	real/complex/hat and an array that contains NE, SE, SW, NW	   *
	* 																   *
	*******************************************************************/
	
	ptype *Host_W_coarse, *Host_W_fine, *Dev_W_coarse, *Dev_W_fine, *Dev_W_H_fine_real;
	
	Host_W_coarse = new ptype[Grid_coarse.N];
	cudaMalloc((void**)&Dev_W_coarse, Grid_coarse.sizeNReal);
	
	Host_W_fine = new ptype[Grid_fine.N];
	cudaMalloc((void**)&Dev_W_fine, Grid_fine.sizeNReal);
	
	//vorticity hermite 
	cudaMalloc((void**)&Dev_W_H_fine_real, 4*Grid_fine.sizeNReal);
	
	
	/*******************************************************************
	*							DISCRET								   *
	*******************************************************************/
	
	#ifdef DISCRET	
		
		ptype *Host_W_initial, *Dev_W_H_initial;
		Host_W_initial = new ptype[Grid_fine.N];		
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
		
		
		//			Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis
		
		
	#endif
	
	
	/*******************************************************************
	*					    Vorticity laplacian						   *
	*******************************************************************/
	/*
	ptype *Host_lap_fine, *Dev_lap_fine_real; 
	cuPtype *Dev_lap_fine_complex, *Dev_lap_fine_hat;
	
	Host_lap_fine = new ptype[Grid_fine.N];
	cudaMalloc((void**)&Dev_lap_fine_real,  Grid_fine.sizeNReal);
	cudaMalloc((void**)&Dev_lap_fine_complex,  Grid_fine.sizeNComplex);
	cudaMalloc((void**)&Dev_lap_fine_hat,  Grid_fine.sizeNComplex);
	*/
	
	/*******************************************************************
	*							  Psi								   *
	* 	Psi is an array that contains Psi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative 							   *
	* 																   *
	*******************************************************************/
	
	//stream hermite on coarse computational grid
	ptype *Host_Psi, *Dev_Psi_real, *Dev_Psi_real_previous;	
	
	Host_Psi = new ptype[4*Grid_coarse.N];
	cudaMalloc((void**) &Dev_Psi_real, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**) &Dev_Psi_real_previous, 4*Grid_coarse.sizeNReal);
	
	
	/*******************************************************************
	*						Gradient of Chi							   *
	* 	We use the gradient of Chi to be sure that the flow is 	       *
	* 	still incompressible 										   *
	* 																   *
	*******************************************************************/
	
	ptype w_min, w_max;
	
	ptype grad_chi_min, grad_chi_max;
	//ptype *Host_gradChi, *Dev_gradChi;
	ptype *Dev_gradChi;
	
	//Host_gradChi = new ptype[Grid_fine.N];
	cudaMalloc((void**) &Dev_gradChi, Grid_fine.sizeNReal);
	
	int grad_block = 32, grad_thread = 1024;
	ptype *Host_w_min, *Host_w_max;
	ptype *Dev_w_min, *Dev_w_max;
	Host_w_min = new ptype[grad_block*grad_thread];
	Host_w_max = new ptype[grad_block*grad_thread];
	cudaMalloc((void**) &Dev_w_min, sizeof(ptype)*grad_block*grad_thread);
	cudaMalloc((void**) &Dev_w_max, sizeof(ptype)*grad_block*grad_thread);
	
	
	/*******************************************************************
	*							 Particles							   *
	*******************************************************************/
	
	int Nb_Tau_p = 20;
	ptype Tau_p[Nb_Tau_p] = {0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.25, 0.5, 0.75, 1, 2, 5, 13};
	int Nb_particles = 1024 * 64; // *512 ; // pow(2, 10)*2048 = 2076672
	int particle_thread =  256 /*pow(2, 10);*/, particle_block = Nb_particles / particle_thread; // 128 ;
	printf("Nb_particles : %d\n", Nb_particles);
	ptype *Host_particles_pos ,*Dev_particles_pos, *Host_particles_vel ,*Dev_particles_vel;
	Host_particles_pos = new ptype[2*Nb_particles*Nb_Tau_p];
	Host_particles_vel = new ptype[2*Nb_particles*Nb_Tau_p];
	cudaMalloc((void**) &Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));
	cudaMalloc((void**) &Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p);
	
	for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
		cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(ptype), cudaMemcpyDeviceToDevice);

	int Nb_fine_dt_particles = 100;
	int freq_fine_dt_particles = 10000;
	int prod_fine_dt_particles = Nb_fine_dt_particles * freq_fine_dt_particles;
	ptype *Dev_particles_pos_fine_dt, *Host_particles_pos_fine_dt;

	Host_particles_pos_fine_dt = new ptype[2*prod_fine_dt_particles];
	cudaMalloc((void**) &Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(ptype));
	
	for(int index_tau_p = 0; index_tau_p < Nb_Tau_p; index_tau_p+=1){
		Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
		//Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
		Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
		Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
	}


	/*******************************************************************
	*				 Energy, Enstrophy, Palinstrophy				   *
	*******************************************************************/

	// Placeholder : Compute initial energy
	int count_mesure = 0;
	ptype *Mesure;
	cudaMallocManaged(&Mesure, 3*tf*sizeof(ptype));
	Compute_Energy<<<1,1024>>>(Mesure, Dev_Psi_real, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	//Compute_Enstrophy<<<1,1024>>>(&(E+1), W, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	//Compute_Palinstrophy<<<1,1024>>>(&(E+2), W, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	cudaDeviceSynchronize();
	printf("ENERGY : %lf, Enstrophy : %lf \n", 0.5 * (*Mesure), 0.5 * (*(Mesure + 1)));
	
	
	ptype *Dev_W_coarse_real_dx_dy;
	cuPtype *Dev_W_coarse_complex_dx_dy, *Dev_W_coarse_hat_dx_dy;
	
	cudaMalloc((void**)&Dev_W_coarse_real_dx_dy, Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_W_coarse_complex_dx_dy, Grid_coarse.sizeNComplex);
	cudaMalloc((void**)&Dev_W_coarse_hat_dx_dy, Grid_coarse.sizeNComplex);
	
	
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
	*						   Initialization						   *
	*******************************************************************/
		
	//initialization of diffeo
	kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiDualX, Dev_ChiDualY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	//cudaDeviceSynchronize();
	
	//setting initial conditions
	translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, map_stack_ctr, Dev_W_fine, Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);

	//in step 1, Psi_p stays the same old one
	evaulate_stream_hermite(&Grid_coarse, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real_previous, cufftPlan_coarse, Dev_Complex_coarse, Dev_Hat_coarse, Dev_Hat_coarse_bis);
	
	cout<<cudaGetErrorName (cudaGetLastError());
	printf("\n");
	
	char buffer[50];
	
	//copying data on host
	cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	
	
	/////////////////////// slight different from regular loop
	//saving max and min for plotting purpose
	get_max_min(&Grid_fine, Host_W_fine, &w_min, &w_max);
	cout<<"W min = "<<w_min<<endl<<"W max = "<<w_max<<endl;
	
	ptype t = t0;
	int loop_ctr = 0;
	int save_ctr = 1;
	
	int old_ctr = 0;

	//int count_fine_dt  = 0;
	
	cout<<std::setprecision(30)<<"dt = "<<dt<<endl;
	
	#ifdef TIME_TESTING
	
		cout<<"Starting time test...\n";
		clock_t begin = clock();
		
	#endif

	cout<<cudaGetErrorName (cudaGetLastError());
	printf("\n");


	/*******************************************************************
	*							 Main loop							   *
	*******************************************************************/
	
	while(tf - t > 0.0000000001 && loop_ctr < iterMax)
	{
		//avoiding over-stepping
		if(t + dt > tf)
			dt = tf - t;
			
		//stream hermite
		evaulate_stream_hermite(&Grid_coarse, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, Dev_Complex_coarse, Dev_Hat_coarse, Dev_Hat_coarse_bis);

		// Particles advection
		//Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
		for(int index_tau_p = 0; index_tau_p < Nb_Tau_p; index_tau_p+=1)
			Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
		//cudaMemcpy(&Dev_particles_pos_fine_dt[(loop_ctr * Nb_fine_dt_particles * 2) % (2*prod_fine_dt_particles)], Dev_particles_pos, 2*Nb_fine_dt_particles*sizeof(ptype), cudaMemcpyDeviceToDevice);
		
		kernel_advect_using_stream_hermite2<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiDualX, Dev_ChiDualY, Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, t, dt, ep);	// time cost		
		kernel_update_map_from_dual<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock, 0, streams[0]>>>(Dev_ChiX, Dev_ChiY, Dev_ChiDualX, Dev_ChiDualY, Grid_coarse.NX, Grid_coarse.NY, ep);																	

	
		//copy Psi to Psi_previous
		//cudaMemcpy(Dev_Psi_real_previous, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
		cudaMemcpyAsync(Dev_Psi_real_previous, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[1]);
		
		
		/*******************************************************************
		*							 Remapping							   *
		*******************************************************************/
		
		grad_chi_min = 1;
		grad_chi_max = 1;
		//incompressibility check (port it on cuda)
		if(loop_ctr % 10 == 0){
			kernel_incompressibility_check<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_gradChi, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);								// time cost		A optimiser
			// We don't need to have Dev_gradChi in memory we juste need to know if it exist a value such as : abs(this_value - 1) > inCompThreshold
			
			//cudaDeviceSynchronize();
			Dev_get_max_min<<<grad_block, grad_thread>>>(Grid_fine.N, Dev_gradChi, Dev_w_min, Dev_w_max);
			//cudaMemcpy(Host_w_min, Dev_w_min, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost);
			//cudaMemcpy(Host_w_max, Dev_w_max, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(Host_w_min, Dev_w_min, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[0]);
			cudaMemcpyAsync(Host_w_max, Dev_w_max, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[1]);
			
			grad_chi_min = Host_w_min[0];
			grad_chi_max = Host_w_max[0];
			for(int i=0;i<grad_block*grad_thread;i++)
			{			
				if(grad_chi_min > Host_w_min[i])
					grad_chi_min = Host_w_min[i];
					
				if(grad_chi_max < Host_w_max[i])
					grad_chi_max = Host_w_max[i];
			}
		}

		//resetting map and adding to stack
		if( ( fabs(grad_chi_min - 1) > inCompThreshold ) || ( fabs(grad_chi_max - 1) > inCompThreshold ) )
		{
			if(map_stack_ctr > map_stack_length)
			{
				cout<<"Stack Saturated... Exiting .. \n";
				break;
			}
			
			#ifndef TIME_TESTING
				printf("Refining Map... ctr = %d \t map_stack_ctr = %d \t gap = %d\n", loop_ctr, map_stack_ctr, loop_ctr - old_ctr); 
				snprintf(logger.buffer, sizeof(logger.buffer), "Refining Map... ctr = %d \t map_stack_ctr = %d \t gap = %d", loop_ctr, map_stack_ctr, loop_ctr - old_ctr);
				logger.push();
				old_ctr = loop_ctr;
			
				//saving map to file
				
			#endif
			
			//saving map stack on device
			cudaMemcpy(&Dev_ChiX_stack[map_stack_ctr*4*Grid_coarse.N], Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
			cudaMemcpy(&Dev_ChiY_stack[map_stack_ctr*4*Grid_coarse.N], Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
			
			//adjusting initial conditions
			translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, map_stack_ctr, Dev_W_fine, Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
			
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
	       
		/*	if (loop_ctr % freq_fine_dt_particles == 0){                             // look at the first hundred particles for a finer time step

		  std::ostringstream fdt;
		  count_fine_dt += 1;
		  cudaMemcpy(Host_particles_pos_fine_dt, Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(ptype), cudaMemcpyDeviceToHost);
		  //		  cudaMallocManaged(&particles_pos_fine_dt, 200 * sizeof(ptype));
		  //particles_pos_fine_dt = Dev_particles_pos;
		  fdt<<count_fine_dt;
		  printf("Saving Frame finer timestep... ctr = %d \t save_ctr = %d  \t time = %f \n", loop_ctr, save_ctr, t); 
		  writeAllRealToBinaryFile(2*prod_fine_dt_particles, Host_particles_pos_fine_dt, simulationName, "particles_pos_fine_dt_" + fdt.str());
		    		  
		  }*/
				  
			
	      
			if( loop_ctr % save_buffer_count == 0 )
			{
				printf("Saving Image... ctr = %d \t save_ctr = %d  \t time = %f \n", loop_ctr, save_ctr, t); 
				snprintf(logger.buffer, sizeof(logger.buffer), "Saving Image... ctr = %d \t save_ctr = %d  \t time = %f", loop_ctr, save_ctr, t); 
				logger.push();
				
				kernel_apply_map_stack_to_W<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Dev_W_fine, map_stack_ctr, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial);
				
				//copying data on host
				cudaMemcpy(Host_W_coarse, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
				cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
				cudaMemcpy(Host_Psi, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
				cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(ptype), cudaMemcpyDeviceToHost);
				
				//writing to file
				std::ostringstream ss;
				ss<<save_ctr;
				
				writeAllRealToBinaryFile(Grid_coarse.N, Host_W_coarse, simulationName, "w_coarse_" + ss.str());
				writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "w_fine_" + ss.str());
				//writeAllRealToBinaryFile(4*Grid_coarse.N, Host_Psi, simulationName, "Psi_" + ss.str()); 
				writeAllRealToBinaryFile(2*Nb_particles*Nb_Tau_p, Host_particles_pos, simulationName, "particles_pos_" + ss.str());
				
				save_ctr++;
				
				//Energy placeholder : need to be put into a array in order to plot it and see the leak of energy
				Compute_Energy<<<1,1024>>>(&Mesure[3*count_mesure], Dev_Psi_real, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
				Compute_Enstrophy<<<1,1024>>>(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
				//Compute_Palinstrophy(&Grid_coarse, Mesure, Dev_W_coarse_hat, Dev_W_coarse_hat_dx_dy, Dev_W_coarse_complex_dx_dy, Dev_W_coarse_real_dx_dy, cufftPlan_coarse); //change mesure
		        //cudaDeviceSynchronize();
				//*(Mesure + 3*count_mesure) = (*(Mesure + 3*count_mesure)) * 0.5 ;
				//*(Mesure + 3*count_mesure +1)) = (*(Mesure + 3*count_mesure +1))) * 0.5;
		        //printf("ENERGY : %lf, Enstrophy: %lf\n", *(Mesure + 3*count_mesure), *(Mesure + 3*count_mesure +1));
				count_mesure+=1;
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
	*					  Zoom on the last frame					   *
	*******************************************************************/
	
	Zoom(simulationName, LX, &Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Dev_W_fine, map_stack_ctr, Dev_W_H_initial);		
	
	/*******************************************************************
	*						 Save final step						   *
	*******************************************************************/
	
	writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "w_final");
	writeAllRealToBinaryFile(4*Grid_coarse.N, Host_Psi, simulationName, "Psi_final");
	writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles_pos_final");

	writeAllRealToBinaryFile(3*tf, Mesure, simulationName, "Mesure");
	
	
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
	
	int cudaError = cudaGetLastError();
	ofstream errorLogFile("data/errorLog.csv", ios::out | ios::app);
	sprintf(buffer, "%e", maxError);
	errorLogFile<<NX_coarse<<", "<<fabs(dt)<<", "<<tf<<", "<<buffer<<","<<cudaError<<endl;
	errorLogFile.close();
	
	
	
	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/
	
	
	
	// Chi
	delete [] Host_ChiX;
	delete [] Host_ChiY;
	cudaFree(Dev_ChiX);
	cudaFree(Dev_ChiY);
	cudaFree(Dev_ChiDualX);
	cudaFree(Dev_ChiDualY);
	
	// Chi_stack
	delete [] Host_ChiX_stack;
	delete [] Host_ChiY_stack;
	cudaFree(Dev_ChiX_stack);
	cudaFree(Dev_ChiY_stack);
	
	// Vorticity
	delete [] Host_W_coarse;
	delete [] Host_W_fine;
	cudaFree(Dev_W_coarse);
	cudaFree(Dev_W_fine);
	cudaFree(Dev_W_H_fine_real);
	
	// Psi
	delete [] Host_Psi;
	cudaFree(Dev_Psi_real);
	cudaFree(Dev_Psi_real_previous);
	
	// Gradient of Chi
	cudaFree(Dev_gradChi);
	
	// CuFFT plans
	cufftDestroy(cufftPlan_coarse);
	cufftDestroy(cufftPlan_fine);
 
	// Particles
	delete [] Host_particles_pos;
	delete [] Host_particles_vel;
	delete [] Host_particles_pos_fine_dt;
	cudaFree(Dev_particles_pos);
	cudaFree(Dev_particles_vel);
	cudaFree(Dev_particles_pos_fine_dt);
	
	

	
	cout<<"Finished; Last Cuda Error : "<<cudaError<<endl;
}


/*******************************************************************
*							 Remapping							   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX_stack, ptype *Dev_ChiY_stack, ptype *Dev_ChiX, ptype *Dev_ChiY, int stack_length, ptype *W_real, ptype *W_H_real, cufftHandle cufftPlan_fine, ptype *W_initial, cuPtype *Dev_Complex_fine, cuPtype *Dev_Hat_fine, cuPtype *Dev_Hat_fine_bis)
{
	
	// Vorticity on coarse grid to vorticity on fine grid
	kernel_apply_map_stack_to_W<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, W_real, stack_length, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
	
	kernel_real_to_complex<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(W_real, Dev_Complex_fine, Grid_fine->NX, Grid_fine->NY);
	cufftExecZ2Z(cufftPlan_fine, Dev_Complex_fine, Dev_Hat_fine, CUFFT_FORWARD);
	kernel_normalize<<<Grid_fine->blocksPerGrid, Grid_fine->threadsPerBlock>>>(Dev_Hat_fine, Grid_fine->NX, Grid_fine->NY);
	
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
	
void evaulate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX, ptype *Dev_ChiY, ptype *Dev_W_H_fine_real, ptype *W_real, ptype *Psi_real, cufftHandle cufftPlan_coarse, cuPtype *Dev_Complex_coarse, cuPtype *Dev_Hat_coarse, cuPtype *Dev_Hat_coarse_bis)
{
	
	// Obtaining stream function at time t 
	kernel_apply_map_and_sample_from_hermite<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, W_real, Dev_W_H_fine_real, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h); 
	kernel_real_to_complex<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Complex_coarse, Dev_Hat_coarse_bis, CUFFT_FORWARD);	
	kernel_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Grid_coarse->NX, Grid_coarse->NY);
	
	// Forming Psi hermite
	kernel_fft_iLap<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Dev_Hat_coarse, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);												// Inverse laplacian in Fourier space
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_coarse, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Psi_real, Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	
	kernel_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse, Dev_Hat_coarse_bis, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);													// y-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_coarse_bis, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(&Psi_real[2*Grid_coarse->N], Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	
	kernel_fft_dx<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse, Dev_Hat_coarse_bis, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);													// x-derivative of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_coarse_bis, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(&Psi_real[Grid_coarse->N], Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	
	kernel_fft_dy<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Dev_Hat_coarse, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h);													// y-derivative of x-derivative of of the vorticity in Fourier space
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_coarse, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(&Psi_real[3*Grid_coarse->N], Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	
}


/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

string create_directory_structure(string simulationName, int NX_coarse, int NXf, double dt, double T, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length, double inCompThreshold)
{
	if (stat("data", &st) == -1) 
	{
		cout<<"A\n";
		mkdir("data", 0700);
	}
	
	//simulationName = simulationName + "_" + currentDateTime();		// Attention !
	//simulationName = simulationName + "_currentDateTime";				
	simulationName = simulationName;
	string folderName = "data/" + simulationName;
	
	//creating folder
	mkdir(folderName.c_str(), 0700);
	
	string folderName1 = folderName + "/all_save_data";
	mkdir(folderName1.c_str(), 0700);
	
	string fileName = folderName + "/notes.txt";
	ofstream file(fileName.c_str(), ios::out);
	
	if(!file)
	{
		cout<<"Error writting files"<<fileName<<endl;
		exit(0);
	}
	else
	{
		file<<"NXc \t\t:"<<NX_coarse<<endl;
		file<<"NXf \t\t:"<<NX_coarse<<endl;
		file<<"dt \t\t:"<<dt<<endl;
		file<<"T \t\t:"<<T<<endl;
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
	
	
	ptype LX;				
	int NXc, NYc;														
	int NXsf, NYsf;														
	int map_stack_ctr = 23;									// don't need it, it can be tertemined by the size of data loaded...
	
	LX = twoPI;	
	NXc = NYc = grid_scale;
	NXsf = NYsf = fine_grid_scale;
	
	string simulationName = File;
	
	TCudaGrid2D Gc(NXc, NYc, LX);
	TCudaGrid2D Gsf(NXsf, NYsf, LX);
	
	
	ptype *ChiX, *ChiY, *ChiX_stack, *ChiY_stack;
	ChiX = new ptype[4*grid_scale*grid_scale];
	ChiY = new ptype[4*grid_scale*grid_scale];
	ChiX_stack = new ptype[map_stack_ctr * 4*Grid_coarse.sizeNReal];	
	ChiY_stack = new ptype[map_stack_ctr * 4*Grid_coarse.sizeNReal];	
	
	
	readAllRealFromBinaryFile(4*grid_scale*grid_scale, ChiX, simulationName, "ChiX_" + t_nb);
	readAllRealFromBinaryFile(4*grid_scale*grid_scale, ChiY, simulationName, "ChiY_" + t_nb);
	readAllRealFromBinaryFile(map_stack_ctr * 4*grid_scale*grid_scale, ChiX_stack, simulationName, "ChiX_stack_" + t_nb);
	readAllRealFromBinaryFile(map_stack_ctr * 4*grid_scale*grid_scale, ChiY_stack, simulationName, "ChiY_stack_" + t_nb);
	
	
	ptype *Dev_W_fine;
	cudaMalloc((void**)&Dev_W_fine,  Grid_fine.sizeNReal);
	
	ptype *Dev_ChiX, *Dev_ChiY;
	cudaMalloc((void**)&Dev_ChiX, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY, 4*Grid_coarse.sizeNReal);
	
	ptype *Dev_ChiX_stack, *Dev_ChiY_stack;
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



























