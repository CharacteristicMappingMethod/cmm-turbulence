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
	int save_buffer_count;												// iterations after which files should be saved
	int map_stack_length;												// this parameter is set to avoide memory overflow on GPU
	int show_progress_at;
	ptype inCompThreshold = 1e-4;										// the maximum allowance of map to deviate from grad_chi begin 1
	
	//GPU dependent parameters
	int mem_RAM_GPU_remaps = 128; 												// mem_index in MB on the GPU
	int mem_RAM_CPU_remaps = 8096;													// mem_RAM_CPU_remaps in MB on the CPU
	int Nb_array_RAM = 4;												// fixed for four different stacks
	double ref_coarse_grid = 128;										// ref value to be compared for stack copzing to cpu, height still unclear
	
	//initialization of parameters
	simulationName = problem_name;
	LX = twoPI;	
	NX_coarse = NY_coarse = grid_scale;
	NX_fine = NY_fine = fine_grid_scale;
	t0 = 0.0;

	// time steps per second used by 4_nodes
	ptype tmp_4nodes = 64;

	/*
	 *  Initial conditions
	 *  "4_nodes" 				-	flow containing exactly 4 fourier modes with two vortices
	 *  "quadropole"			-	???
	 *  "three_vortices"		-	???
	 *  "single_shear_layer"	-	shear layer problem forming helmholtz-instabilities, merging into two vortices which then merges into one big vortex
	 *  "two_vortices"			-	???
	 *  "turbulence_gaussienne"	-	???
	 */
	// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"
	if(simulationName == "4_nodes")
	{
		grid_by_time = 1.0;
		snapshots_per_second = 1;
		dt = 1.0/ tmp_4nodes;//(NX_coarse * grid_by_time);
		tf = 2;
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
		tf = 50; //50;//0.5;//100;//300;//
	}
	else if(simulationName == "two_votices")
	{
		grid_by_time = 8.0;
		snapshots_per_second = 1;
		dt = 1.0/(NX_coarse * grid_by_time);
		tf = 10;		
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
		ptype *Dev_W_H_initial;
		cudaMalloc((void**)&Dev_W_H_initial, 8);
	#endif
	
	
	//shared parameters
	iterMax = ceil(tf / dt);
	save_buffer_count = tmp_4nodes/snapshots_per_second;//(NX_coarse * grid_by_time) / snapshots_per_second;//	tmp/snapshots_per_second;//(NX_coarse * grid_by_time) / snapshots_per_second;							// the denominator is snapshots per second
	show_progress_at = (32 * 4 * pow(ref_coarse_grid, 3.0)) / pow(NX_coarse, 3.0);
		if(show_progress_at < 1) show_progress_at = 1;
	map_stack_length = (mem_RAM_GPU_remaps * pow(ref_coarse_grid, 2.0))/ (double(NX_coarse * NX_coarse));
	int frac_mem_cpu_to_gpu = int(double(mem_RAM_CPU_remaps)/double(mem_RAM_GPU_remaps)/double(Nb_array_RAM));  // define how many more remappings we can save on CPU than on GPU
	
	cout<<"Simulation name : "<<simulationName<<endl;
	cout<<"Iter max : "<<iterMax<<endl;
	cout<<"Save buffer count : "<<save_buffer_count<<endl;
	cout<<"Progress at : "<<show_progress_at<<endl;
	cout<<"map stack length : "<<map_stack_length<<endl;
	cout<<"map stack length on RAM : "<<frac_mem_cpu_to_gpu * map_stack_length<<endl;
	cout<<"map stack length total on RAM : "<<frac_mem_cpu_to_gpu * map_stack_length * Nb_array_RAM<<endl;
	
	// create naming for folder structure, dependent on time integration
	string sim_name_addition;
	if (TIME_INTEGRATION == "EulerExp") {
		sim_name_addition = "_DEV_EulExp_";
	}
	else if (TIME_INTEGRATION == "ABTwo") {
		sim_name_addition = "_DEV_AB2_";
	}
	else if (TIME_INTEGRATION == "RKThree") {
		sim_name_addition = "_ULYSSE_";
	}
	else sim_name_addition = "_DEV_UNNOWN_";

	simulationName = simulationName + sim_name_addition + std::to_string(grid_scale) + "_" + std::to_string(tmp_4nodes).substr(0, std::to_string(tmp_4nodes).find(".")); //"vortex_shear_1000_4";
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
	// TCudaGrid2D Grid_NDFT(128, 128, LX);
	
	
	/*******************************************************************
	*							CuFFT plans							   *
	* 	Plan use to compute FFT using Cuda library CuFFT	 	       *
	* 																   *
	*******************************************************************/
	
	cufftHandle cufftPlan_coarse, cufftPlan_fine;
	cufftPlan2d(&cufftPlan_coarse, Grid_coarse.NX, Grid_coarse.NY, CUFFT_Z2Z);
	cufftPlan2d(&cufftPlan_fine, Grid_fine.NX, Grid_fine.NY, CUFFT_Z2Z);
	
	
	/*******************************************************************
	*							Trash variable	   					   *
	*******************************************************************/
	
	cuPtype *Dev_Complex_fine, *Dev_Hat_fine, *Dev_Hat_fine_bis;
	cudaMalloc((void**)&Dev_Complex_fine, Grid_fine.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_fine, Grid_fine.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_fine_bis, Grid_fine.sizeNComplex);
	
	
	/*******************************************************************
	*								Test NDFT						   *
	*******************************************************************/
	
	/*
	printf("NDFT\n");
	
	int Np_particles = 16384;
	int iNDFT_block, iNDFT_thread = 256;
	iNDFT_block = Np_particles/256;
	int *f_k, *Dev_f_k;
	ptype *x_1_n, *x_2_n, *p_n, *Dev_x_1_n, *Dev_x_2_n, *Dev_p_n, *X_k_bis;
	cuPtype *X_k, *Dev_X_k, *Dev_X_k_derivative;
	
	x_1_n = new ptype[Np_particles];
	x_2_n = new ptype[Np_particles];
	p_n = new ptype[2*Np_particles];
	f_k = new int[Grid_NDFT.NX];
	X_k = new cuPtype[Grid_NDFT.N];
	X_k_bis = new ptype[2*Grid_NDFT.N];
	cudaMalloc((void**)&Dev_x_1_n, sizeof(ptype)*Np_particles);
	cudaMalloc((void**)&Dev_x_2_n, sizeof(ptype)*Np_particles);
	cudaMalloc((void**)&Dev_p_n, sizeof(ptype)*2*Np_particles);
	cudaMalloc((void**)&Dev_f_k, sizeof(int)*Np_particles);
	cudaMalloc((void**)&Dev_X_k, Grid_NDFT.sizeNComplex);
	cudaMalloc((void**)&Dev_X_k_derivative, Grid_NDFT.sizeNComplex);
	
	readRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x1.data");
	readRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x2.data");
	readRealToBinaryAnyFile(2*Np_particles, p_n, "src/Initial_W_discret/p.data");
	
	for(int i = 0; i < Grid_NDFT.NX; i+=1)
		f_k[i] = i;
	
	cudaMemcpy(Dev_x_1_n, x_1_n, sizeof(ptype)*Np_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_x_2_n, x_2_n, sizeof(ptype)*Np_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_p_n, p_n, sizeof(ptype)*2*Np_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_f_k, f_k, sizeof(int)*Grid_NDFT.NX, cudaMemcpyHostToDevice);
	
	printf("NDFT v_x\n");
	NDFT_2D<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX, Np_particles);
	printf("iNDFT v_x\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_1_n, Dev_x_1_n, sizeof(ptype)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_1_n, "src/Initial_W_discret/x_1_ifft.data");
	
	printf("kernel_fft_dx\n");
	kernel_fft_dx<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
	printf("iNDFT v_x/dx\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k_derivative, Dev_x_1_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_1_n, Dev_x_1_n, sizeof(ptype)*Np_particles, cudaMemcpyDeviceToHost);
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
	cudaMemcpy(x_2_n, Dev_x_2_n, sizeof(ptype)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x_2_ifft.data");
	
	printf("kernel_fft_dy\n");
	kernel_fft_dy<<<Grid_NDFT.blocksPerGrid, Grid_NDFT.threadsPerBlock>>>(Dev_X_k, Dev_X_k_derivative, Grid_NDFT.NX, Grid_NDFT.NY, Grid_NDFT.h);
	printf("iNDFT v_y/dy\n");
	iNDFT_2D<<<iNDFT_block, iNDFT_thread>>>(Dev_X_k_derivative, Dev_x_2_n, Dev_p_n, Dev_f_k, Grid_NDFT.NX);
	cudaMemcpy(x_2_n, Dev_x_2_n, sizeof(ptype)*Np_particles, cudaMemcpyDeviceToHost);
	writeRealToBinaryAnyFile(Np_particles, x_2_n, "src/Initial_W_discret/x_2_dy_ifft.data");
	
	printf("Fini NDFT\n");
	*/
	
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
	
	ptype *Host_ChiX_stack_RAM_0, *Host_ChiY_stack_RAM_0, *Host_ChiX_stack_RAM_1, *Host_ChiY_stack_RAM_1, *Host_ChiX_stack_RAM_2, *Host_ChiY_stack_RAM_2, *Host_ChiX_stack_RAM_3, *Host_ChiY_stack_RAM_3, *Dev_ChiX_stack, *Dev_ChiY_stack;
	
	cudaMalloc((void **) &Dev_ChiX_stack, map_stack_length * 4*Grid_coarse.sizeNReal);	
	cudaMalloc((void **) &Dev_ChiY_stack, map_stack_length * 4*Grid_coarse.sizeNReal);
	int map_stack_ctr = 0;
	cout<<"Map Stack Initialized"<<endl;
	logger.push("Map Stack Initialized");
	
	int stack_length_RAM = -1;
	int stack_length_Nb_array_RAM = -1;
	Host_ChiX_stack_RAM_0 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_0 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiX_stack_RAM_1 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_1 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiX_stack_RAM_2 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_2 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiX_stack_RAM_3 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	Host_ChiY_stack_RAM_3 = new ptype[frac_mem_cpu_to_gpu * map_stack_length * 4*Grid_coarse.sizeNReal];
	
	
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
		
	#endif
	
	/*******************************************************************
	*							  Psi								   *
	* 	Psi is an array that contains Psi, x1-derivative,		       *
	* 	x2-derivative and x1x2-derivative 							   *
	* 																   *
	*******************************************************************/
	
	//stream hermite on coarse computational grid
    ptype *Host_Psi, *Dev_Psi_real, *Dev_Psi_real_previous, *Dev_Psi_real_previous_p;

	Host_Psi = new ptype[4*Grid_coarse.N];
	cudaMalloc((void**) &Dev_Psi_real, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**) &Dev_Psi_real_previous, 4*Grid_coarse.sizeNReal);
	cudaMalloc((void**) &Dev_Psi_real_previous_p, 4*Grid_coarse.sizeNReal);

	
	/*******************************************************************
	*						Gradient of Chi							   *
	* 	We use the gradient of Chi to be sure that the flow is 	       *
	* 	still incompressible 										   *
	* 																   *
	*******************************************************************/
	
	ptype w_min, w_max;
	
	ptype grad_chi_min, grad_chi_max;
	
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
	
    #ifdef PARTICLES

	   	// int Nb_Tau_p = 21;
		// ptype Tau_p[Nb_Tau_p] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.25, 0.5, 0.75, 1, 2, 5, 13};
	   	int Nb_Tau_p = 1;
		ptype Tau_p[Nb_Tau_p] = {0.0};
		int Nb_particles = 1024 * 8; //64; // *512 ;
        int particle_thread =  256 /*pow(2, 10);*/, particle_block = Nb_particles / particle_thread; // 128 ;
        printf("Nb_particles : %d\n", Nb_particles);
        ptype *Host_particles_pos ,*Dev_particles_pos, *Host_particles_vel ,*Dev_particles_vel;
        Host_particles_pos = new ptype[2*Nb_particles*Nb_Tau_p];
        Host_particles_vel = new ptype[2*Nb_particles*Nb_Tau_p];
        cudaMalloc((void**) &Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));
        cudaMalloc((void**) &Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));

        ptype *Dev_particles_pos_1, *Dev_particles_vel_1,*Dev_particles_pos_2, *Dev_particles_vel_2;
        cudaMalloc((void**) &Dev_particles_pos_1, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));
        cudaMalloc((void**) &Dev_particles_vel_1, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));
        cudaMalloc((void**) &Dev_particles_pos_2, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));
        cudaMalloc((void**) &Dev_particles_vel_2, 2*Nb_particles*Nb_Tau_p*sizeof(ptype));



       /* #ifdef RKThree_PARTICLES
            ptype *Dev_particles_vel_previous, *Dev_particles_vel_previous_p;
            cudaMalloc((void**) &Dev_particles_vel_previous, 2*Nb_particles*Nb_Tau_p*sizeof(ptype)); // Might be too big to carry around three instant of the velocity
            cudaMalloc((void**) &Dev_particles_vel_previous_p, 2*Nb_particles*Nb_Tau_p*sizeof(ptype)); // with Nb_particles = 1024*512, these 3 arrays take ~503MB on the graphics card
        #endif*/

        curandGenerator_t prng;
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniformDouble(prng, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p);

        cudaMemcpy(Dev_particles_pos_1, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy(Dev_particles_pos_2, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p, cudaMemcpyDeviceToDevice);

        for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
            cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(ptype), cudaMemcpyDeviceToDevice);

        int Nb_fine_dt_particles = 1000;
        int freq_fine_dt_particles = save_buffer_count; // redundant
        int prod_fine_dt_particles = Nb_fine_dt_particles * freq_fine_dt_particles;
        ptype *Dev_particles_pos_fine_dt, *Host_particles_pos_fine_dt;

        Host_particles_pos_fine_dt = new ptype[2*prod_fine_dt_particles];
        cudaMalloc((void**) &Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(ptype));

       /* Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
        Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, Dev_particles_pos);
        #pragma unroll Nb_Tau_p - 1
        for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
            Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
            //Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
            Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

            #ifndef RKThree_PARTICLES
                Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
            #else
                //Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel_previous[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
                //Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel_previous_p[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

               // Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
            #endif
        }*/

    #endif


	/*******************************************************************
	*				 ( Measure and file organization )				   *
	*******************************************************************/

	int count_mesure = 0;
	int mes_size = tf*snapshots_per_second;
    ptype *Mesure;
    ptype *Mesure_fine;
	cudaMallocManaged(&Mesure, 3*mes_size*sizeof(ptype));
	cudaMallocManaged(&Mesure_fine, 3*mes_size*sizeof(ptype));


    // File organization : Might be moved
    std::string fi, element[4] = {"particles", "vorticity_coarse", "vorticity_fine", "Stream_function"};
    for(int i = 0; i<4; i+=1){
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
        fi = "data/" + simulationName + "/all_save_data/vorticity_fine_lagrangian";
        mkdir(fi.c_str(), 0700);


    // Laplacian
	/*
    ptype *Host_lap_fine, *Dev_lap_fine_real;
    cuPtype *Dev_lap_fine_complex, *Dev_lap_fine_hat;

    Host_lap_fine = new ptype[Grid_fine.N];

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
	*	    Variable Nicolas(Convergence and psi upsampling) 	   	   *
	*******************************************************************/
	
	
	TCudaGrid2D Grid_2048(1024, 1024, LX);
	
	ptype *Host_2048_4;
	ptype *Dev_ChiX_2048, *Dev_ChiY_2048, *Dev_W_2048;
	
	Host_2048_4 = new ptype[4*Grid_2048.N];
	cudaMalloc((void**)&Dev_ChiX_2048, Grid_2048.sizeNReal);
	cudaMalloc((void**)&Dev_ChiY_2048, Grid_2048.sizeNReal);
	
	cudaMalloc((void**)&Dev_W_2048, Grid_2048.sizeNReal);
	
	cufftHandle cufftPlan_2048;
	cufftPlan2d(&cufftPlan_2048, Grid_2048.NX, Grid_2048.NY, CUFFT_Z2Z);
	
	
	cuPtype *Dev_Complex_fine_2048, *Dev_Hat_fine_2048, *Dev_Hat_fine_bis_2048;
	cudaMalloc((void**)&Dev_Complex_fine_2048, Grid_2048.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_fine_2048, Grid_2048.sizeNComplex);
	cudaMalloc((void**)&Dev_Hat_fine_bis_2048, Grid_2048.sizeNComplex);
	
    ptype *Dev_Psi_2048, *Dev_Psi_2048_previous, *Dev_Psi_2048_previous_p;
    
	cudaMalloc((void**) &Dev_Psi_2048, 4*Grid_2048.sizeNReal);
    cudaMalloc((void**) &Dev_Psi_2048_previous, 4*Grid_2048.sizeNReal);
    cudaMalloc((void**) &Dev_Psi_2048_previous_p, 4*Grid_2048.sizeNReal);

    // Laplacian

    ptype *Host_lap_fine_2048, *Dev_lap_fine_2048_real;
    cuPtype *Dev_lap_fine_2048_complex, *Dev_lap_fine_2048_hat;

    Host_lap_fine_2048 = new ptype[Grid_2048.N];

    cudaMalloc((void**)&Dev_lap_fine_2048_real, Grid_2048.sizeNReal);
    cudaMalloc((void**)&Dev_lap_fine_2048_complex, Grid_2048.sizeNComplex);
    cudaMalloc((void**)&Dev_lap_fine_2048_hat, Grid_2048.sizeNComplex);




	/*******************************************************************
	*						   Initialization						   *
	*******************************************************************/
		
	//initialization of diffeo
	kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	//kernel_init_diffeo<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiDualX, Dev_ChiDualY, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	//cudaDeviceSynchronize();
	
	//setting initial conditions
	translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Dev_W_fine, Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
	
	//in step 1, Psi_p stays the same old one

    evaulate_stream_hermite(&Grid_coarse, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real_previous, cufftPlan_coarse, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);


    //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);


     //  cudaMemcpy(Dev_Psi_real_previous_p_p, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
    //  cudaDeviceSynchronize();

    cudaMemcpy(Dev_Psi_real_previous_p, Dev_Psi_real_previous, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
   /* cudaMemcpy(Dev_Psi_2048_previous_p, Dev_Psi_2048_previous, 4*Grid_2048.sizeNReal, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();*/

	char buffer[50];
	
	//copying data on host
    cudaMemcpy(Host_Psi, Dev_Psi_real_previous, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(4*Grid_coarse.N, Host_Psi, simulationName, "Stream_function/Psi_0");


	cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "vorticity_fine/w_fine_0");
	cudaMemcpy(Host_W_coarse, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);	
	writeAllRealToBinaryFile(Grid_coarse.N, Host_W_coarse, simulationName, "vorticity_coarse/w_coarse_0");
    cudaDeviceSynchronize();

    kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial);
    cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
    writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_0");

    //Laplacian initial

    Laplacian_vort(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_2048, Dev_lap_fine_2048_real, Dev_lap_fine_2048_complex, Dev_lap_fine_2048_hat, cufftPlan_2048);

    cudaMemcpy(Host_lap_fine_2048, Dev_lap_fine_2048_real, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    writeAllRealToBinaryFile(Grid_2048.N, Host_lap_fine_2048, simulationName, "vorticity_fine_lagrangian/w_lagr_0");


    // They're everywhere ! need function

    kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial);
	cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "w_2048_0");

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





    // Compute initial quantities (L2 norm)
    Compute_Energy<<<Grid_coarse.blocksPerGrid,Grid_coarse.threadsPerBlock>>>(&Mesure[3*count_mesure], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
	Compute_Enstrophy<<<Grid_coarse.blocksPerGrid,Grid_coarse.threadsPerBlock>>>(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
    Compute_Palinstrophy(&Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, cufftPlan_coarse);

    //Compute_Energy<<<Grid_2048.blocksPerGrid,Grid_2048.threadsPerBlock>>>(&Mesure_fine[3*count_mesure], Dev_Psi_2048, Grid_2048.N, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
    Compute_Enstrophy<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Mesure_fine[1 + 3*count_mesure], Dev_W_fine, Grid_fine.N, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);
    Compute_Palinstrophy(&Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, cufftPlan_fine);

    count_mesure+=1;

    #ifdef PARTICLES

        // Particles initialization
        Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, Dev_particles_pos);
        Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, Dev_particles_pos_1);
        Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, Dev_particles_pos_2);

        Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
        Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt/2, Dev_particles_pos_1, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
        Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt/4, Dev_particles_pos_2, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

        //Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, Dev_particles_pos);
        #pragma unroll Nb_Tau_p - 1
        for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
            Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, &Dev_particles_pos[2*Nb_particles*index_tau_p]);
            Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, &Dev_particles_pos_1[2*Nb_particles*index_tau_p]);
            Rescale<<<particle_block, particle_thread>>>(Nb_particles, twoPI, &Dev_particles_pos_2[2*Nb_particles*index_tau_p]);

            //Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
            Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
            Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt/2, &Dev_particles_pos_1[2*Nb_particles*index_tau_p], &Dev_particles_vel_1[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
            Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt/4, &Dev_particles_pos_2[2*Nb_particles*index_tau_p], &Dev_particles_vel_2[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

            #ifndef RKThree_PARTICLES
                Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt/2, &Dev_particles_pos_1[2*Nb_particles*index_tau_p], &Dev_particles_vel_1[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt/4, &Dev_particles_pos_2[2*Nb_particles*index_tau_p], &Dev_particles_vel_2[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);

                // Deux fois Dev_psi_real_previous car Dev_Psi_real n'est pas encore initialisé                                                                                                                ^^           ^^
            #else
                //Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel_previous[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
                //Particle_advect_iner_ini<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel_previous_p[2*Nb_particles*index_tau_p], Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

                Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt/2, &Dev_particles_pos_1[2*Nb_particles*index_tau_p], &Dev_particles_vel_1[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt/4, &Dev_particles_pos_2[2*Nb_particles*index_tau_p], &Dev_particles_vel_2[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);

            #endif
            }
    #endif




	/////////////////////// slight different from regular loop
	//saving max and min for plotting purpose
	get_max_min(&Grid_fine, Host_W_fine, &w_min, &w_max);
	cout<<"W min = "<<w_min<<endl<<"W max = "<<w_max<<endl;
	
	ptype t = t0;
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
		evaulate_stream_hermite(&Grid_coarse, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_coarse, Dev_Psi_real, cufftPlan_coarse, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);

        //evaulate_stream_hermite(&Grid_2048, &Grid_fine, Dev_ChiX, Dev_ChiY, Dev_W_H_fine_real, Dev_W_2048, Dev_Psi_2048_previous, cufftPlan_2048, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);


		// Particles advection
        #ifdef PARTICLES
            Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt, Dev_particles_pos, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
            cudaMemcpy(&Dev_particles_pos_fine_dt[(loop_ctr * Nb_fine_dt_particles * 2) % (2*prod_fine_dt_particles)], Dev_particles_pos, 2*Nb_fine_dt_particles*sizeof(ptype), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt/2, Dev_particles_pos_1, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
            cudaDeviceSynchronize();
            Particle_advect<<<particle_block, particle_thread>>>(Nb_particles, dt/4, Dev_particles_pos_2, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);

            #pragma unroll Nb_Tau_p - 1
            for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1){
                #ifndef RKThree_PARTICLES
                     Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                     Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt/2, &Dev_particles_pos_1[2*Nb_particles*index_tau_p], &Dev_particles_vel_1[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                     Particle_advect_inertia<<<particle_block, particle_thread>>>(Nb_particles, dt/4, &Dev_particles_pos_2[2*Nb_particles*index_tau_p], &Dev_particles_vel_2[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);

                #else
                    Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt, &Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_vel[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                    Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt/2, &Dev_particles_pos_1[2*Nb_particles*index_tau_p], &Dev_particles_vel_1[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);
                    Particle_advect_inertia_RK3<<<particle_block, particle_thread>>>(Nb_particles, dt/4, &Dev_particles_pos_2[2*Nb_particles*index_tau_p], &Dev_particles_vel_2[2*Nb_particles*index_tau_p], Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Tau_p[index_tau_p]);

                #endif

            }
        #endif

        kernel_advect_using_stream_hermite2<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiDualX, Dev_ChiDualY, Dev_Psi_real, Dev_Psi_real_previous, Dev_Psi_real_previous_p, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, t, dt, ep);	// time cost

        //kernel_advect_using_stream_hermite2<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, Dev_ChiDualX, Dev_ChiDualY, Dev_Psi_2048, Dev_Psi_2048_previous, Dev_Psi_2048_previous_p, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, t, dt, ep);	// time cost
        kernel_update_map_from_dual<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock, 0, streams[0]>>>(Dev_ChiX, Dev_ChiY, Dev_ChiDualX, Dev_ChiDualY, Grid_coarse.NX, Grid_coarse.NY, ep);


       /* #ifdef RKThree_PARTICLES
            // copy particles velocity to previous instance
           cudaMemcpy(Dev_particles_vel_previous_p, Dev_particles_vel_previous, 2*Nb_particles*Nb_Tau_p*sizeof(ptype), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(Dev_particles_vel_previous, Dev_particles_vel, 2*Nb_particles*Nb_Tau_p*sizeof(ptype), cudaMemcpyDeviceToDevice);
        #endif*/

        //copy Psi to Psi_previous and Psi_previous to Psi_previous_previous
		//cudaMemcpy(Dev_Psi_real_previous, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
        cudaMemcpy(Dev_Psi_real_previous_p, Dev_Psi_real_previous, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(Dev_Psi_real_previous, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToDevice, streams[1]);
		

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
			// compute minimum for actual check on dev, coppy to machine
			Dev_get_max_min<<<grad_block, grad_thread>>>(Grid_fine.N, (cufftDoubleReal*)Dev_Complex_fine, Dev_w_min, Dev_w_max);// Dev_gradChi cufftDoubleComplex cufftDoubleReal
			//cudaMemcpy(Host_w_min, Dev_w_min, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost);
			//cudaMemcpy(Host_w_max, Dev_w_max, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(Host_w_min, Dev_w_min, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[0]);
			cudaMemcpyAsync(Host_w_max, Dev_w_max, sizeof(ptype)*grad_block*grad_thread, cudaMemcpyDeviceToHost, streams[1]);  // not needed?
			
			grad_chi_min = Host_w_min[0];
			grad_chi_max = Host_w_max[0];  // not needed?
			// for-loop to compute final min/max on host
			for(int i=0;i<grad_block*grad_thread;i++)
			{			
				if(grad_chi_min > Host_w_min[i])
					grad_chi_min = Host_w_min[i];
					
				if(grad_chi_max < Host_w_max[i])  // not needed?
					grad_chi_max = Host_w_max[i];  // not needed?
			}
		}
			
		//resetting map and adding to stack
		if( ( fabs(grad_chi_min - 1) > inCompThreshold ) || ( fabs(grad_chi_max - 1) > inCompThreshold ) )
		{
			if(map_stack_ctr > map_stack_length*frac_mem_cpu_to_gpu*Nb_array_RAM)
			{
				cout<<"Stack Saturated... Exiting .. \n";
				break;
			}
			
			#ifndef TIME_TESTING
				printf("Refining Map... ctr = %d \t map_stack_ctr = %d ; %d ; %d \t gap = %d \t incomp_err = %e\n", loop_ctr, map_stack_ctr, stack_length_RAM, stack_length_Nb_array_RAM, loop_ctr - old_ctr, fmax(fabs(grad_chi_min - 1), fabs(grad_chi_max - 1)));
				snprintf(logger.buffer, sizeof(logger.buffer), "Refining Map... ctr = %d \t map_stack_ctr = %d \t gap = %d \t incomp_err = %e", loop_ctr, map_stack_ctr, loop_ctr - old_ctr, fmax(fabs(grad_chi_min - 1), fabs(grad_chi_max - 1)));
				logger.push();
				old_ctr = loop_ctr;
			#endif
			
			//adjusting initial conditions
			translate_initial_condition_through_map_stack(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_ChiX, Dev_ChiY, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Dev_W_fine, Dev_W_H_fine_real, cufftPlan_fine, Dev_W_H_initial, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis);
			
			
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



           if(loop_ctr == 110 || loop_ctr == 126){
                kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial);
                cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
                writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_" + std::to_string(loop_ctr));

            }

			if( loop_ctr % save_buffer_count == 0 )
			{
			  printf("Saving Image... ctr = %d \t save_ctr = %d  \t time = %f \t  Compute time : %lf \n", loop_ctr, save_ctr, t, double(clock()-begin)/CLOCKS_PER_SEC);
				snprintf(logger.buffer, sizeof(logger.buffer), "Saving Image... ctr = %d \t save_ctr = %d  \t time = %f", loop_ctr, save_ctr, t); 
				logger.push();
				
				//copying data on host
				cudaMemcpy(Host_W_coarse, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
				cudaMemcpy(Host_Psi, Dev_Psi_real, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
                #ifdef PARTICLES
                    cudaMemcpy(Host_particles_pos_fine_dt, Dev_particles_pos_fine_dt, 2*prod_fine_dt_particles*sizeof(ptype), cudaMemcpyDeviceToHost);
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
                            cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*Nb_particles*Nb_Tau_p*sizeof(ptype), cudaMemcpyDeviceToHost);
                            //cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles/fluid/particles_pos_" + ss.str());
                            /*cudaMemcpy(Host_particles_pos, Dev_particles_pos_1, 2*Nb_particles*Nb_Tau_p*sizeof(ptype), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles/fluid/particles_pos_dt_" + ss.str());
                            cudaMemcpy(Host_particles_pos, Dev_particles_pos_2, 2*Nb_particles*Nb_Tau_p*sizeof(ptype), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles/fluid/particles_pos_dt2_" + ss.str());
			    */
                            for(int i = 1; i < Nb_Tau_p; i+=1)
                                writeAllRealToBinaryFile(2*Nb_particles, &Host_particles_pos[i * 2*Nb_particles], simulationName, "particles/" + std::to_string(Tau_p[i]).substr(0, std::to_string(Tau_p[i]).find(".") + 3+ 1) + "/particles_pos_" + ss.str());
                        }
                    }

                #endif
				if (save_ctr%1==0){
					kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_fine, Dev_Complex_fine, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial);
                    cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
                    writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "vorticity_fine/w_fine_" + ss.str());

                    kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_2048, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_2048, Dev_Complex_fine_2048, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_2048.NX, Grid_2048.NY, Grid_2048.h, Dev_W_H_initial);
                    cudaMemcpy(Host_2048_4, Dev_W_2048, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
                    writeAllRealToBinaryFile(Grid_2048.N, Host_2048_4, simulationName, "vorticity_fine/w_1024_" + ss.str());


                   /* kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_plot, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_plot, Dev_Complex_plot, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_plot.NX, Grid_plot.NY, Grid_plot.h, Dev_W_H_initial);

                    cudaMemcpy(Host_W_plot, Dev_W_plot, Grid_plot.sizeNReal, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    writeAllRealToBinaryFile(Grid_plot.N, Host_W_plot, simulationName, "vorticity_fine/w_plot_" + ss.str());*/
				}
				/*
				if (save_ctr%50==0){
					for(int index_tau_p = 1; index_tau_p < Nb_Tau_p; index_tau_p+=1)
						cudaMemcpy(&Dev_particles_pos[2*Nb_particles*index_tau_p], &Dev_particles_pos[0], 2*Nb_particles*sizeof(ptype), cudaMemcpyDeviceToDevice);
				}
	            */
				save_ctr++;

                //Mesure on the coarse grid
                Compute_Energy<<<Grid_coarse.blocksPerGrid,Grid_coarse.threadsPerBlock>>>(&Mesure[3*count_mesure], Dev_Psi_real, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
				Compute_Enstrophy<<<Grid_coarse.blocksPerGrid, Grid_coarse.threadsPerBlock>>>(&Mesure[1 + 3*count_mesure], Dev_W_coarse, Grid_coarse.N, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h);
				Compute_Palinstrophy(&Grid_coarse, &Mesure[2 + 3*count_mesure], Dev_W_coarse, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, cufftPlan_coarse);

                //Measure on the fine grid
                //Compute_Energy<<<Grid_2048.blocksPerGrid,Grid_2048.threadsPerBlock>>>(&Mesure_fine[3*count_mesure], Dev_Psi_2048, Grid_2048.N, Grid_2048.NX, Grid_2048.NY, Grid_2048.h);
                Compute_Enstrophy<<<Grid_fine.blocksPerGrid, Grid_fine.threadsPerBlock>>>(&Mesure_fine[1 + 3*count_mesure], Dev_W_fine, Grid_fine.N, Grid_fine.NX, Grid_fine.NY, Grid_fine.h);
                Compute_Palinstrophy(&Grid_fine, &Mesure_fine[2 + 3*count_mesure], Dev_W_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_Hat_fine_bis, cufftPlan_fine);

				count_mesure+=1;

                //Laplacian_vort(&Grid_fine, Dev_W_fine, Dev_Complex_fine, Dev_Hat_fine, Dev_lap_fine_real, Dev_lap_fine_complex, Dev_lap_fine_hat, cufftPlan_fine);

                //cudaMemcpy(Host_lap_fine, Dev_lap_fine_real, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
                //cudaDeviceSynchronize();
                //writeAllRealToBinaryFile(Grid_fine.N, Host_lap_fine, simulationName, "vorticity_fine_lagrangian/w_lagr_" + ss.str());


                 //Laplacian initial

                Laplacian_vort(&Grid_2048, Dev_W_2048, Dev_Complex_fine_2048, Dev_Hat_fine_2048, Dev_lap_fine_2048_real, Dev_lap_fine_2048_complex, Dev_lap_fine_2048_hat, cufftPlan_2048);
                cudaMemcpy(Host_lap_fine_2048, Dev_lap_fine_2048_real, Grid_2048.sizeNReal, cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                writeAllRealToBinaryFile(Grid_2048.N, Host_lap_fine_2048, simulationName, "vorticity_fine_lagrangian/w_lagr_" + ss.str());


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
	
	kernel_apply_map_stack_to_W_part_All(&Grid_coarse, &Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, Dev_W_fine, Dev_Complex_fine, map_stack_ctr, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, frac_mem_cpu_to_gpu, Grid_coarse.NX, Grid_coarse.NY, Grid_coarse.h, Grid_fine.NX, Grid_fine.NY, Grid_fine.h, Dev_W_H_initial);
	cudaMemcpy(Host_W_fine, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(Grid_fine.N, Host_W_fine, simulationName, "w_fine_final");
	writeAllRealToBinaryFile(4*Grid_coarse.N, Host_Psi, simulationName, "Psi_final");
    #ifdef PARTICLES
        writeAllRealToBinaryFile(2*Nb_particles, Host_particles_pos, simulationName, "particles_pos_final");
    #endif
	writeAllRealToBinaryFile(3*mes_size, Mesure, simulationName, "Mesure");
	writeAllRealToBinaryFile(3*mes_size, Mesure_fine, simulationName, "Mesure_fine");
	
	
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
	
	int cudaError = cudaGetLastError();
	ofstream errorLogFile("data/errorLog.csv", ios::out | ios::app);
	sprintf(buffer, "%e", maxError);
	errorLogFile<<NX_coarse<<", "<<fabs(dt)<<", "<<tf<<", "<<buffer<<","<<cudaError<<endl;
	errorLogFile.close();
	
	
	
	/*******************************************************************
	*						 Freeing memory							   *
	*******************************************************************/
	
	
	// Trash variable
	cudaFree(Dev_Complex_fine);
	cudaFree(Dev_Hat_fine);
	cudaFree(Dev_Hat_fine_bis);
	
	// Chi
	delete [] Host_ChiX;
	delete [] Host_ChiY;
	cudaFree(Dev_ChiX);
	cudaFree(Dev_ChiY);
	//cudaFree(Dev_ChiDualX);
	//cudaFree(Dev_ChiDualY);
	
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
	
	// Psi
	delete [] Host_Psi;
	cudaFree(Dev_Psi_real);
	cudaFree(Dev_Psi_real_previous);
	
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
	cout<<"Finished; Last Cuda Error : "<<cudaError<<endl;
}


/*******************************************************************
*							 Remapping							   *
*******************************************************************/

void translate_initial_condition_through_map_stack(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX_stack, ptype *Dev_ChiY_stack, ptype *Host_ChiX_stack_RAM_0, ptype *Host_ChiY_stack_RAM_0, ptype *Host_ChiX_stack_RAM_1, ptype *Host_ChiY_stack_RAM_1, ptype *Host_ChiX_stack_RAM_2, ptype *Host_ChiY_stack_RAM_2, ptype *Host_ChiX_stack_RAM_3, ptype *Host_ChiY_stack_RAM_3, ptype *Dev_ChiX, ptype *Dev_ChiY, int stack_length, int map_stack_length, int stack_length_RAM, int stack_length_Nb_array_RAM, int mem_RAM, ptype *W_real, ptype *W_H_real, cufftHandle cufftPlan_fine, ptype *W_initial, cuPtype *Dev_Complex_fine, cuPtype *Dev_Hat_fine, cuPtype *Dev_Hat_fine_bis)
{
	
	// Vorticity on coarse grid to vorticity on fine grid
	kernel_apply_map_stack_to_W_part_All(Grid_coarse, Grid_fine, Dev_ChiX_stack, Dev_ChiY_stack, Dev_ChiX, Dev_ChiY, Host_ChiX_stack_RAM_0, Host_ChiY_stack_RAM_0, Host_ChiX_stack_RAM_1, Host_ChiY_stack_RAM_1, Host_ChiX_stack_RAM_2, Host_ChiY_stack_RAM_2, Host_ChiX_stack_RAM_3, Host_ChiY_stack_RAM_3, W_real, Dev_Complex_fine, stack_length, map_stack_length, stack_length_RAM, stack_length_Nb_array_RAM, mem_RAM, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h, W_initial);
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
	
void evaulate_stream_hermite(TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_fine, ptype *Dev_ChiX, ptype *Dev_ChiY, ptype *Dev_W_H_fine_real, ptype *W_real, ptype *Psi_real, cufftHandle cufftPlan_coarse, cuPtype *Dev_Complex_coarse, cuPtype *Dev_Hat_coarse, cuPtype *Dev_Hat_coarse_bis)
{
	
	// Obtaining stream function at time t 
	kernel_apply_map_and_sample_from_hermite<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_ChiX, Dev_ChiY, W_real, Dev_W_H_fine_real, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_coarse->NX, Grid_coarse->NY, Grid_coarse->h, Grid_fine->NX, Grid_fine->NY, Grid_fine->h); 
	kernel_real_to_complex<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Complex_coarse, Dev_Hat_coarse_bis, CUFFT_FORWARD);	
	kernel_normalize<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Grid_coarse->NX, Grid_coarse->NY);
	
	cut_off_scale<<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(Dev_Hat_coarse_bis, Grid_coarse->NX);
	cufftExecZ2Z(cufftPlan_coarse, Dev_Hat_coarse_bis, Dev_Complex_coarse, CUFFT_INVERSE);
	kernel_complex_to_real  <<<Grid_coarse->blocksPerGrid, Grid_coarse->threadsPerBlock>>>(W_real, Dev_Complex_coarse, Grid_coarse->NX, Grid_coarse->NY);
	
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

string create_directory_structure(string simulationName, int NX_coarse, int NX_fine, double dt, double T, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length, double inCompThreshold)
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
        #ifdef RKThree
            file<<"Time integration : RK3"<<endl;
        #elif defined(ABTwo)
              file<<"Time integration : AB2"<<endl;
        #else
              file<<"Time integration : Euler explicit"<<endl;
        #endif

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



























