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

#include "settings.h"

#include "../grid/cmm-grid2d.h"  // for PI and twoPI

void SettingsCMM::setPresets() {
	// naming and saving settings of the simulation
	std::string workspace = "./"; // where should the files be saved? "./" or "" means at the run location, has to end with backslash
	std::string sim_name = "debug";  // unique identifier to differentiate simulations

	// grid settings for coarse and fine grid
	// 	8		16		32		64		128		256		512		1024	2048	4096	8192	16384	32768
	// max working on Anthicythere : grid_scale = 8192; fine_grid_scale = 16384;
	int grid_coarse = 512;
	int grid_fine = 2048;
	int grid_psi = 1024;  // psi will be (up)sampled on this grid, Restriction: 2*N_fft_psi !> 4*N_coarse
	int grid_vort = grid_fine;  // vorticity will be sampled on this grid for computation of psi, this changes the scales of the vorticity

	/*
	 *  Initial conditions for vorticity
	 *  "4_nodes" 				-	flow containing exactly 4 fourier modes with two vortices
	 *  "quadropole"			-	???
	 *  "one_vortex"			-	one vortex in center, stationary flow for investigations
	 *  "two_vortices"			-	two vortices of same sign which merge after a while
	 *  "three_vortices"		-	two vortices of same sign with one of opposing sign for more complex dynamics
	 *  "single_shear_layer"	-	shear layer problem forming helmholtz-instabilities, merging into two vortices which then merges into one big vortex
	 *  "tanh_shear_layer"		-	shear layer with tanh boundary
	 *  "turbulence_gaussienne"	-	gaussian blobs in checkerboard - version made by thibault
	 *  "gaussian_blobs"		-	gaussian blobs in checkerboard - version made by julius
	 *  "shielded_vortex"		-	vortex core with ring of negative vorticity around it
	 *  "two_cosine"			-	stationary setup of two cosine vortices
	 *  "vortex_sheets"			-	vortex sheets used for singularity studies
	 */
	std::string initial_condition = "4_nodes";
	// string containing parameters for initial condition, they have to be formatted to fit {D1,D2,D3,...} or "D1,D2,D3,..."
	// can be 10 values max, check cmm-init.cu if the initial condition needs parameter
	std::string initial_params = "{0}";

	// possibility to compute from discrete initial condition
	bool initial_discrete = false;
	int initial_discrete_grid = 2048;  // variable gridsize for discrete initial grid
	std::string initial_discrete_location = "data/final-state-anticythere-1_gaussian_blobs_C1024_F8192_t3.07e+03_T500/Time_data/Time_270/Vorticity_W_4096.data";  // path to discrete file, relative from workspace

	/*
	 * Console output verbose intensity
	 *  0	-	no console outputs besides errors
	 *  1	-	Initialization and finishing output
	 *  2	-	Step and Saving output
	 *  3	-	Version stuff and Conservation results
	 *  4	-	Array initialization details
	 */
	int verbose = 3;

	// set time properties
	double final_time = 3;  // end of computation
	bool set_dt_by_steps = true;  // choose whether we want to set dt by steps or by grid
	double factor_dt_by_grid = 1;  // if dt is set by the grid (cfl), then this should be the max velocity
	int steps_per_sec = 32;  // how many steps do we want per seconds?
	// dt will be set in cmm-euler, so that all changes can be applied there

	/*
	 * Which variables do we want to save? Best separated by a "-"
	 * Vorticity: "Vorticity", "W"
	 *
	 * Stream function: "Stream", "Psi"
	 * Stream function hermite: "Stream_H", "Psi_H" - not for zoom
	 * Velocity: "Velocity", "U" - not for zoom
	 *
	 * Backwards map: "Map_b", "Chi_b"
	 * Backwards map Hermite: "Map_H", "Chi_H" - only for computational
	 *
	 * Laplacian of vorticity: "Laplacian_W" - only for sample - is stream function?
	 * Gradient of vorticity: "Grad_W" - only for sample
	 *
	 * Passive scalar: "Scalar", "Theta" - not for computational_var
	 * Advected Particles: "PartA_XX" - not for sample_var, XX is the particle computation number
	 * Advected Particles velocity: "PartA_Vel_XX" - only for computational_var, XX is the particle computation number
	 *
	 * Forward map: "Map_f", "Chi_f"
	 * Forwards map Hermite: "Map_f_H", "Chi_f_H" - only for computational
	 * Forwarded Particles: "PartF_XX" - not for computational_var, XX is the particle computation number
	 */
	// time instants or intervals at what we want to save computational data, 0 for initial and T_MAX for final
	int save_computational_num = 3;
	std::string save_computational_s[12] = {
			"{is_instant=1,time_start=0,var=W-U,conv=1}",  // save begin
			"{is_instant=1,time_start="+str_t(T_MAX)+",var=W-U,conv=1}",  // save end
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=0.5,var ,conv=1}",  // conv over simulation
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_01,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_02,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_03,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_04,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_05,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_06,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_07,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_08,conv=0}",
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=PartA_Vel_01,conv=0}"
	};


	// set minor properties
	double incomp_threshhold = 1e-5;  // the maximum allowance of map to deviate from grad_chi begin 1
	double map_epsilon = 1e-3;  // distance used for foot points for GALS map advection
	// skip remapping, useful for convergence tests
	bool skip_remapping = false;

	// set memory properties
	int mem_RAM_CPU_remaps = 9000;  // mem_RAM_CPU_remaps in MB on the CPU
	bool save_map_stack = false;  // possibility to save the map stack to reuse for other computations to skip initial time
	// restart simulation
	double restart_time = 0;  // other than zero means the simulation is restarted
	std::string restart_location = "";  // if empty, then data is read from own data folder


	// set specific settings
	/*
	 * Time integration
	 * First order: "EulerExp" (or "RK1")
	 * Second order: "AB2", "RK2" (or "Heun")
	 * Third order: "RK3", "RK3Mod"
	 * Fourth order: "RK4", "RK4Mod"
	 */
	std::string time_integration = "RK3";
	/*
	 * Override lagrange interpolation for velocity
	 * if -1, then lagrange order is set after time integration schemes
	 * values range between 1 and 4 implementation-wise
	 * 1 does not work for RK3Mod by definition
	 * works in general not with particles
	 */
	int lagrange_override = -1;
	bool lagrange_init_higher_order = true;  // initialization with EulerExp or increasing order?

	// mapupdate order, "2nd", "4th", "6th"
	std::string map_update_order = "4th";

	// mollification settings, stencil size, 0, 4, 8
	int molly_stencil = 0;

	// in addition to the upsampling, we want to lowpass in fourier space by cutting high frequencies
	double freq_cut_psi = (double)(grid_coarse)/4.0;  // take into account, that frequencies are symmetric around N/2

	// time instants or intervals at what we want to save computational data, 0 for initial and T_MAX for final
	int save_sample_num = 1;
	std::string save_sample_s[1] = {
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=W,grid=1024}"  // save over simulation
	};


	/*
	 * Passive scalar settings
	 *  Initial conditions for transport of passive scalar
	 *  "rectangle"					-	simple rectangle with sharp borders
	 *  "gaussian"					-	normal distribution / gaussian blob
	 *  "circular_ring"				-   circular ring around center
	 */
	std::string scalar_name = "rectangle";
	// NOT IMPLEMENTED YET - possibility to compute from discrete initial scalar condition
	bool scalar_discrete = false;
	int scalar_discrete_grid = 2048;  // variable gridsize for discrete initial scalar grid
	std::string scalar_discrete_location = "I-am-not-implemented-yet!";  // path to discrete file, relative from workspace


	/*
	 * Zoom settings
	 */
	// time instants or intervalls at what we want to save computational data, 0 for initial and T_MAX for final
	int save_zoom_num = 0;
	std::string save_zoom_s[2] = {
			"{is_instant=0,time_start=0,time_end="+str_t(T_MAX)+",time_step=1,var=W-PartA_01,grid=1024"
			",pos_x="+str_t(twoPI * 0.5)+",pos_y="+str_t(twoPI * 0.6)+",width_x="+str_t(twoPI * 2e-1)+",width_y="+str_t(twoPI * 2e-1)+
			",rep="+str_t(2)+",rep_fac="+str_t(0.5)+"}",
			"{is_instant=0,time_start=2,time_end="+str_t(T_MAX)+",time_step=1,var=W-Psi,grid=1024"
			",pos_x="+str_t(twoPI * 0.5)+",pos_y="+str_t(twoPI * 0.6)+",width_x="+str_t(twoPI * 2e-1)+",width_y="+str_t(twoPI * 2e-1)+
			",rep="+str_t(2)+",rep_fac="+str_t(0.5)+"}"
	};



	/*
	 * Forward map settings to compute forward map for scalar particles,
	 */
	bool forward_map = false;  // en- or disable computing of forward map

	// forwarded particles, parameters similar to advected particles
	int particles_forwarded_num = 0;
	std::string particles_forwarded_s[2] = {
		"{num=1000000,seed=0,init_name=uniform,init_time=0"
		",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
		"{num=20000,seed=0,init_name=circular_ring,init_time=1"
		",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI/2.0)+",init_param_4="+str_t(PI/2.0)+"}"
	};


	/*
	 * Particle settings for advected particles
	 *
	 * num - amount of particles
	 * tau - inertial strength of particles, 0 for fluid particles
	 * seed - seed for random number generator
	 *
	 * time_integration - Time integration for particles
	 *  First order: "EulerExp"
	 *  Second order: "AB2", "RK2"
	 *  Third order: "RK3", "RK3Mod"
	 *  Fourth order: "RK4", "RK4Mod"
	 *
	 * init_name - Initial conditions for particle position
	 *  "uniform"					-	uniformly distributed in particular frame
	 *  "normal", "gaussian"		-	normal / gaussian distributed around center with specific variance
	 *  "circular_ring"				-   circular ring around specific center with particles_width as radius
	 *  "uniform_grid"				-   uniform grid with equal amount of points in x- and y-direction in particular frame
	 *
	 * init_time - when should the computation for the particles start
	 * init_vel - if the inertial particles velocity should be set after the velocity or to zero
	 * init_param - specific parameters to control the initial condition
	 */
	int particles_advected_num = 2;
	std::string particles_advected_s[8] = {
			"{num=100000,tau=0,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=0.1,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=0.2,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=0.5,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=1,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=1.5,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=2.5,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}",
			"{num=100000,tau=5,seed=0,time_integration=RK3,init_name=uniform,init_time=0,init_vel=0"
			",init_param_1="+str_t(PI)+",init_param_2="+str_t(PI)+",init_param_3="+str_t(PI*2.0)+",init_param_4="+str_t(PI*2.0)+"}"
	};
	int particles_steps = -1;  // hackery for particle convergence


	// make sure that not initialized values are set
	lagrange_order = 0;
	// now set everything
	setWorkspace(workspace); setSimName(sim_name);
	setGridCoarse(grid_coarse); setGridFine(grid_fine);
	setGridPsi(grid_psi); setGridVort(grid_vort);
	setFinalTime(final_time);
	setSetDtBySteps(set_dt_by_steps); setFactorDtByGrid(factor_dt_by_grid); setStepsPerSec(steps_per_sec);

	setSaveComputationalNum(save_computational_num);
	for (int i_save = 0; i_save < save_computational_num; ++i_save) {
		save_computational[i_save] = SaveComputational();
		save_computational[i_save].setAllVariables(save_computational_s[i_save]);
	}

	setInitialCondition(initial_condition);
	setInitialParams(initial_params);
	setInitialDiscrete(initial_discrete);
	setInitialDiscreteGrid(initial_discrete_grid);
	setInitialDiscreteLocation(initial_discrete_location);

	setVerbose(verbose);

	setMemRamCpuRemaps(mem_RAM_CPU_remaps);
	setSaveMapStack(save_map_stack);
	setRestartTime(restart_time);
	setRestartLocation(restart_location);

	setIncompThreshold(incomp_threshhold);
	setMapEpsilon(map_epsilon);
	setTimeIntegration(time_integration);
	setLagrangeOverride(lagrange_override);
	setLagrangeInitHigherOrder(lagrange_init_higher_order);
	setMapUpdateOrder(map_update_order);
	setMollyStencil(molly_stencil);
	setFreqCutPsi(freq_cut_psi);
	setSkipRemapping(skip_remapping);

	setSaveSampleNum(save_sample_num);
	for (int i_save = 0; i_save < save_sample_num; ++i_save) {
		save_sample[i_save] = SaveSample();
		save_sample[i_save].setAllVariables(save_sample_s[i_save]);
	}

	setScalarName(scalar_name);
	setScalarDiscrete(scalar_discrete);
	setScalarDiscreteGrid(scalar_discrete_grid);
	setScalarDiscreteLocation(scalar_discrete_location);

	setSaveZoomNum(save_zoom_num);
	for (int i_save = 0; i_save < save_zoom_num; ++i_save) {
		save_zoom[i_save] = SaveZoom();
		save_zoom[i_save].setAllVariables(save_zoom_s[i_save]);
	}

	setForwardMap(forward_map);
	setParticlesForwardedNum(particles_forwarded_num);
	for (int i_particles = 0; i_particles < particles_forwarded_num; ++i_particles) {
		particles_forwarded[i_particles] = ParticlesForwarded();
		particles_forwarded[i_particles].setAllVariables(particles_forwarded_s[i_particles]);
	}

	setParticlesAdvectedNum(particles_advected_num);
	for (int i_particles = 0; i_particles < particles_advected_num; ++i_particles) {
		particles_advected[i_particles] = ParticlesAdvected();
		particles_advected[i_particles].setAllVariables(particles_advected_s[i_particles]);
	}

	setParticlesSteps(particles_steps);
}


/*
 *  Function to apply values taken from command line
 *  general form:  COMMAND=VALUE
 */
void SettingsCMM::applyCommands(int argc, char *args[]) {
	// loop over all commands
	for( int count = 0; count < argc; count++ ) {
		// construct string for command
		std::string command_full = args[count];
		// set variable, separated by = sign
		setVariable(command_full, "=");
	}
	//	 cout << "  args[" << count << "]   " << args[count] << "\n";
}


// function to apply command with variable delimiter to change it
int SettingsCMM::setVariable(std::string command_full, std::string delimiter) {
	// check for delimiter position
	int pos_equal = command_full.find(delimiter);
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+delimiter.length(), command_full.length());

		// big if else for different commands
		// this beast is becoming larger and larger, i should convert it to something more automatic
		// link to site for that: https://stackoverflow.com/questions/4480788/c-c-switch-case-with-string
		if (command == "workspace") setWorkspace(value);
		else if (command == "sim_name") setSimName(value);
		else if (command == "grid_coarse") setGridCoarse(std::stoi(value));
		else if (command == "grid_fine") setGridFine(std::stoi(value));
		else if (command == "grid_psi") setGridPsi(std::stoi(value));
		else if (command == "grid_vort") setGridVort(std::stoi(value));

		else if (command == "final_time") setFinalTime(std::stod(value));
		else if (command == "factor_dt_by_grid") setFactorDtByGrid(std::stod(value));
		else if (command == "steps_per_sec") setStepsPerSec(std::stoi(value));
		else if (command == "set_dt_by_steps") setSetDtBySteps(getBoolFromString(value));

		else if (command == "save_computational_num") {
			setSaveComputationalNum(std::stoi(value));
			return 1;
		}

		else if (command == "mem_RAM_CPU_remaps") setMemRamCpuRemaps(std::stoi(value));
		else if (command == "save_map_stack") setSaveMapStack(getBoolFromString(value));
		else if (command == "restart_time") setRestartTime(std::stod(value));
		else if (command == "restart_location") setRestartLocation(value);

		else if (command == "verbose") setVerbose(std::stoi(value));

		else if (command == "initial_condition") setInitialCondition(value);
		else if (command == "initial_params") setInitialParams(value);
		else if (command == "initial_discrete") setInitialDiscrete(getBoolFromString(value));
		else if (command == "initial_discrete_grid") setInitialDiscreteGrid(stoi(value));
		else if (command == "initial_discrete_location") setInitialDiscreteLocation(value);

		else if (command == "incomp_threshold") setIncompThreshold(std::stod(value));
		else if (command == "map_epsilon") setMapEpsilon(std::stod(value));
		else if (command == "time_integration") setTimeIntegration(value);
		else if (command == "lagrange_override") setLagrangeOverride(std::stoi(value));
		else if (command == "lagrange_init_higher_order") setLagrangeInitHigherOrder(getBoolFromString(value));
		else if (command == "map_update_order") setMapUpdateOrder(value);
		else if (command == "molly_stencil") setMollyStencil(std::stoi(value));
		else if (command == "freq_cut_psi") setFreqCutPsi(std::stod(value));
		else if (command == "skip_remapping") setSkipRemapping(getBoolFromString(value));

		else if (command == "save_sample_num") {
			setSaveSampleNum(std::stoi(value));
			return 2;
		}

		else if (command == "scalar_name") setScalarName(value);
		else if (command == "scalar_discrete") setScalarDiscrete(getBoolFromString(value));
		else if (command == "scalar_discrete_grid") setScalarDiscreteGrid(stoi(value));
		else if (command == "scalar_discrete_location") setScalarDiscreteLocation(value);

		else if (command == "save_zoom_num") {
			setSaveZoomNum(std::stoi(value));
			return 3;
		}

		else if (command == "forward_map") setForwardMap(getBoolFromString(value));

		else if (command == "particles_forwarded_num") {
			setParticlesForwardedNum(std::stoi(value));
			return 5;
		}

		else if (command == "particles_advected_num") {
			setParticlesAdvectedNum(std::stoi(value));
			return 4;
		}

		// hackery for particle convergence
		else if (command == "particles_steps") setParticlesSteps(std::stoi(value));
	}

	return 0;
}


// class constructor from tfour main ingredients
SettingsCMM::SettingsCMM(std::string sim_name, int gridCoarse, int gridFine, std::string initialCondition) {
	// initially allocate arrays to be able to delete them later
	save_computational = new SaveComputational[1];
	save_sample = new SaveSample[1];
	save_zoom = new SaveZoom[1];
	particles_advected = new ParticlesAdvected[1];
	particles_forwarded = new ParticlesForwarded[1];
	// set presets
	setPresets();
	// override the four main components
	setSimName(sim_name);
	setGridCoarse(gridCoarse);
	setGridFine(gridFine);
	setInitialCondition(initialCondition);
	// assume psi will not be upsampled
	setGridPsi(gridCoarse);
}


// class constructor to build from presets
SettingsCMM::SettingsCMM() {
	// initially allocate arrays to be able to delete them later
	save_computational = new SaveComputational[1];
	save_sample = new SaveSample[1];
	save_zoom = new SaveZoom[1];
	particles_advected = new ParticlesAdvected[1];
	particles_forwarded = new ParticlesForwarded[1];
	// set presets
	setPresets();
}


// class constructor to take into account command line inputs
SettingsCMM::SettingsCMM(int argc, char *args[]) {
	// initially allocate arrays to be able to delete them later
	save_computational = new SaveComputational[1];
	save_sample = new SaveSample[1];
	save_zoom = new SaveZoom[1];
	particles_advected = new ParticlesAdvected[1];
	particles_forwarded = new ParticlesForwarded[1];
	// set presets
	setPresets();
	// override presets with command line arguments
	applyCommands(argc, args);
}


void SettingsCMM::setSaveComputational(std::string command_full, std::string delimiter, int number) {
	// check for delimiter position
	int pos_equal = command_full.find(delimiter);
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+delimiter.length(), command_full.length());

		save_computational[number] = SaveComputational();
		save_computational[number].setAllVariables(value);
	}
}

void SettingsCMM::setSaveSample(std::string command_full, std::string delimiter, int number) {
	// check for delimiter position
	int pos_equal = command_full.find(delimiter);
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+delimiter.length(), command_full.length());

		save_sample[number] = SaveSample();
		save_sample[number].setAllVariables(value);
	}
}

void SettingsCMM::setSaveZoom(std::string command_full, std::string delimiter, int number) {
	// check for delimiter position
	int pos_equal = command_full.find(delimiter);
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+delimiter.length(), command_full.length());

		save_zoom[number] = SaveZoom();
		save_zoom[number].setAllVariables(value);
	}
}

void SettingsCMM::setParticlesAdvected(std::string command_full, std::string delimiter, int number) {
	// check for delimiter position
	int pos_equal = command_full.find(delimiter);
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+delimiter.length(), command_full.length());

		particles_advected[number] = ParticlesAdvected();
		particles_advected[number].setAllVariables(value);

		std::cout << "PartA read in: " << value << "\n";
	}
}

void SettingsCMM::setParticlesForwarded(std::string command_full, std::string delimiter, int number) {
	// check for delimiter position
	int pos_equal = command_full.find(delimiter);
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+delimiter.length(), command_full.length());

		particles_forwarded[number] = ParticlesForwarded();
		particles_forwarded[number].setAllVariables(value);
	}
}



// functions for save computational to save the values we use to compute
void SaveComputational::setAllVariables(std::string s_array) {
	// parse check: erase " or {} if surrounded by it and continue reading values
	if ((s_array.substr(0, 1) == "\"" && s_array.substr(s_array.length()-1, s_array.length()) == "\"") ||
	   (s_array.substr(0, 1) == "{"  && s_array.substr(s_array.length()-1, s_array.length()) == "}")) {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
		// loop over all elements
		int index = 0;
		std::string::size_type pos = 0; std::string::size_type pos_new = 0;
		do {
			pos_new = s_array.find(",", pos);
			std::string substring;
			if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new - pos);
			else substring = s_array.substr(pos, s_array.length());
			setVariable(substring);
			index++;
			pos = pos_new + 1;
		} while (pos_new != std::string::npos);
	}
}
void SaveComputational::setVariable(std::string command_full) {
	// check for delimiter position
	int pos_equal = command_full.find("=");
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+1, command_full.length());

		// if else for different commands
		if (command == "is_instant") is_instant = getBoolFromString(value);
		else if (command == "time_start") time_start = std::stod(value);
		else if (command == "time_end") time_end = std::stod(value);
		else if (command == "time_step") time_step = std::stod(value);
		else if (command == "var") var = value;
		else if (command == "conv") conv = getBoolFromString(value);
	}
}
std::string SaveComputational::getVariables() {
	std::ostringstream os;
	os << "{is_instant=" << str_t(is_instant) << ",time_start=" << str_t(time_start);
	if (!is_instant) { os << ",time_end=" << str_t(time_end) << ",time_step=" << str_t(time_step); }
	os << ",var=" << var << ",conv=" << str_t(conv) << "}";
	return os.str();
}


// functions for save sample to define what we want to sample and when
void SaveSample::setAllVariables(std::string s_array) {
	// parse check: erase " or {} if surrounded by it and continue reading values
	if ((s_array.substr(0, 1) == "\"" && s_array.substr(s_array.length()-1, s_array.length()) == "\"") ||
	   (s_array.substr(0, 1) == "{"  && s_array.substr(s_array.length()-1, s_array.length()) == "}")) {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
		// loop over all elements
		int index = 0;
		std::string::size_type pos = 0; std::string::size_type pos_new = 0;
		do {
			pos_new = s_array.find(",", pos );
			std::string substring;
			if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new - pos);
			else substring = s_array.substr(pos, s_array.length());
			setVariable(substring);
			index++;
			pos = pos_new + 1;
		} while (pos_new != std::string::npos);
	}
}
void SaveSample::setVariable(std::string command_full) {
	// check for delimiter position
	int pos_equal = command_full.find("=");
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+1, command_full.length());

		// if else for different commands
		if (command == "is_instant") is_instant = getBoolFromString(value);
		else if (command == "time_start") time_start = std::stod(value);
		else if (command == "time_end") time_end = std::stod(value);
		else if (command == "time_step") time_step = std::stod(value);
		else if (command == "var") var = value;
		else if (command == "grid") grid = stoi(value);
	}
}
std::string SaveSample::getVariables() {
	std::ostringstream os;
	os << "{is_instant=" << str_t(is_instant) << ",time_start=" << str_t(time_start);
	if (!is_instant) { os << ",time_end=" << str_t(time_end) << ",time_step=" << str_t(time_step); }
	os << ",var=" << var << ",grid=" << str_t(grid) << "}";
	return os.str();
}


// functions for save sample to define what we want to sample and when
void SaveZoom::setAllVariables(std::string s_array) {
	// parse check: erase " or {} if surrounded by it and continue reading values
	if ((s_array.substr(0, 1) == "\"" && s_array.substr(s_array.length()-1, s_array.length()) == "\"") ||
	   (s_array.substr(0, 1) == "{"  && s_array.substr(s_array.length()-1, s_array.length()) == "}")) {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
		// loop over all elements
		int index = 0;
		std::string::size_type pos = 0; std::string::size_type pos_new = 0;
		do {
			pos_new = s_array.find(",", pos );
			std::string substring;
			if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new - pos);
			else substring = s_array.substr(pos, s_array.length());
			setVariable(substring);
			index++;
			pos = pos_new + 1;
		} while (pos_new != std::string::npos);
	}
}
void SaveZoom::setVariable(std::string command_full) {
	// check for delimiter position
	int pos_equal = command_full.find("=");
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+1, command_full.length());

		// if else for different commands
		if (command == "is_instant") is_instant = getBoolFromString(value);
		else if (command == "time_start") time_start = std::stod(value);
		else if (command == "time_end") time_end = std::stod(value);
		else if (command == "time_step") time_step = std::stod(value);
		else if (command == "var") var = value;
		else if (command == "grid") grid = stoi(value);
		else if (command == "pos_x") pos_x = stod(value);
		else if (command == "pos_y") pos_y = stod(value);
		else if (command == "width_x") width_x = stod(value);
		else if (command == "width_y") width_y = stod(value);
		else if (command == "rep") rep = stoi(value);
		else if (command == "rep_fac") rep_fac = stod(value);
	}
}
std::string SaveZoom::getVariables() {
	std::ostringstream os;
	os << "{is_instant=" << str_t(is_instant) << ",time_start=" << str_t(time_start);
	if (!is_instant) { os << ",time_end=" << str_t(time_end) << ",time_step=" << str_t(time_step); }
	os << ",var=" << var << ",grid=" << str_t(grid);
	os << ",pos_x=" << str_t(pos_x) << ",pos_y=" << pos_y << ",width_x=" << str_t(width_x) << ",width_y=" << str_t(width_y);
	os << ",rep=" << str_t(rep) << ",rep_fac=" << str_t(rep_fac) << "}";
	return os.str();
}


// functions to define advected particles which are tracked
void ParticlesAdvected::setAllVariables(std::string s_array) {
	// parse check: erase " or {} if surrounded by it and continue reading values
	if ((s_array.substr(0, 1) == "\"" && s_array.substr(s_array.length()-1, s_array.length()) == "\"") ||
	   (s_array.substr(0, 1) == "{"  && s_array.substr(s_array.length()-1, s_array.length()) == "}")) {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
		// loop over all elements
		int index = 0;
		std::string::size_type pos = 0; std::string::size_type pos_new = 0;
		do {
			pos_new = s_array.find(",", pos );
			std::string substring;
			if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new - pos);
			else substring = s_array.substr(pos, s_array.length());
			setVariable(substring);
			index++;
			pos = pos_new + 1;
		} while (pos_new != std::string::npos);
	}
}
void ParticlesAdvected::setVariable(std::string command_full) {
	// check for delimiter position
	int pos_equal = command_full.find("=");
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+1, command_full.length());

		// if else for different commands
		if (command == "num") num = std::stoi(value);
		else if (command == "tau") tau = std::stod(value);
		else if (command == "seed") seed = std::stoull(value);
		else if (command == "time_integration") {
			time_integration = value;
			if (time_integration == "EulerExp") { time_integration_num = 10; }
			else if (time_integration == "RK1") { time_integration_num = 10; }
			else if (time_integration == "Heun") { time_integration_num = 20; }
			else if (time_integration == "RK2") { time_integration_num = 20; }
			else if (time_integration == "RK3") { time_integration_num = 30; }
			else if (time_integration == "RK4") { time_integration_num = 40; }
			else if (time_integration == "RK3Mod") { time_integration_num = 31; }
			else if (time_integration == "RK4Mod") { time_integration_num = 41; }
			else time_integration_num = -1;
		}
		else if (command == "init_name") {
			init_name = value;
			if(init_name == "uniform") init_num = 0;
			else if(init_name == "normal" or init_name == "gaussian") init_num = 1;
			else if(init_name == "circular_ring") init_num = 2;
			else if(init_name == "uniform_grid") init_num = 3;
			else if(init_name == "sine_sheets") init_num = 4;
			else init_num = -1;
		}
		else if (command == "init_time") init_time = stod(value);
		else if (command == "init_vel") init_vel = getBoolFromString(value);
		else if (command == "init_param_1") init_param_1 = stod(value);
		else if (command == "init_param_2") init_param_2 = stod(value);
		else if (command == "init_param_3") init_param_3 = stod(value);
		else if (command == "init_param_4") init_param_4 = stod(value);
	}
}
std::string ParticlesAdvected::getVariables() {
	std::ostringstream os;
	os << "{num=" << str_t(num) << ",tau=" << str_t(tau) << ",seed=" << str_t(seed);
	os << ",time_integration=" << time_integration << ",init_name=" << init_name << ",init_time=" << str_t(init_time) << ",init_vel=" << str_t(init_vel);
	os << ",init_param_1=" << str_t(init_param_1) << ",init_param_2=" << str_t(init_param_2) << ",init_param_3=" << str_t(init_param_3) << ",init_param_4=" << str_t(init_param_4) << "}";
	return os.str();
}


// functions to define advected particles which are tracked
void ParticlesForwarded::setAllVariables(std::string s_array) {
	// parse check: erase " or {} if surrounded by it and continue reading values
	if ((s_array.substr(0, 1) == "\"" && s_array.substr(s_array.length()-1, s_array.length()) == "\"") ||
	   (s_array.substr(0, 1) == "{"  && s_array.substr(s_array.length()-1, s_array.length()) == "}")) {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
		// loop over all elements
		int index = 0;
		std::string::size_type pos = 0; std::string::size_type pos_new = 0;
		do {
			pos_new = s_array.find(",", pos );
			std::string substring;
			if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new - pos);
			else substring = s_array.substr(pos, s_array.length());
			setVariable(substring);
			index++;
			pos = pos_new + 1;
		} while (pos_new != std::string::npos);
	}
}
void ParticlesForwarded::setVariable(std::string command_full) {
	// check for delimiter position
	int pos_equal = command_full.find("=");
	if (pos_equal != std::string::npos) {
		// construct two substrings
		std::string command = command_full.substr(0, pos_equal);
		std::string value = command_full.substr(pos_equal+1, command_full.length());

		// if else for different commands
		if (command == "num") num = std::stoi(value);
		else if (command == "seed") seed = std::stoull(value);
		else if (command == "init_name") {
			init_name = value;
			if(init_name == "uniform") init_num = 0;
			else if(init_name == "normal" or init_name == "gaussian") init_num = 1;
			else if(init_name == "circular_ring") init_num = 2;
			else if(init_name == "uniform_grid") init_num = 3;
			else if(init_name == "sine_sheets") init_num = 4;
			else init_num = -1;
		}
		else if (command == "init_time") init_time = stod(value);
		else if (command == "init_param_1") init_param_1 = stod(value);
		else if (command == "init_param_2") init_param_2 = stod(value);
		else if (command == "init_param_3") init_param_3 = stod(value);
		else if (command == "init_param_4") init_param_4 = stod(value);
	}
}
std::string ParticlesForwarded::getVariables() {
	std::ostringstream os;
	os << "{num=" << str_t(num) << ",seed=" << str_t(seed) << ",init_name=" << init_name << ",init_time=" << str_t(init_time);
	os << ",init_param_1=" << str_t(init_param_1) << ",init_param_2=" << str_t(init_param_2) << ",init_param_3=" << str_t(init_param_3) << ",init_param_4=" << str_t(init_param_4) << "}";
	return os.str();
}


bool getBoolFromString(std::string value) {
	return (value == "true" || value == "True" || value == "1");
//	if (value == "true" || value == "True" || value == "1") return true;
//	else if (value == "false" || value == "False" || value == "0") return false;
//	return false;  // in case no value is chosen
}
