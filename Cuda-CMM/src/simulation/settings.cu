#include "settings.h"

void SettingsCMM::setPresets() {
	// naming and saving settings of the simulation
	string workspace = "./"; // where should the files be saved? "./" or "" means at the run location, has to end with backslash
	string sim_name = "debug";  // unique identifier to differentiate simulations

	// grid settings for coarse and fine grid
	// 32		64		128		256		512		1024		2048		4096		8192		16384
	// max working on V100 : grid_scale = 4096; fine_grid_scale = 16384;
	int grid_coarse = 128;
	int grid_fine = 1024;
	int grid_psi = 2048;  // psi will be used on this grid

	/*
	 *  Initial conditions
	 *  "4_nodes" 				-	flow containing exactly 4 fourier modes with two vortices
	 *  "quadropole"			-	???
	 *  "three_vortices"		-	???
	 *  "single_shear_layer"	-	shear layer problem forming helmholtz-instabilities, merging into two vortices which then merges into one big vortex
	 *  "two_vortices"			-	???
	 *  "turbulence_gaussienne"	-	???
	 */
	string initial_condition = "4_nodes";

	// set time properties
	double final_time = 4;  // end of computation
	double factor_dt_by_grid = 1;  // if dt is set by the grid (cfl), then this is the factor for it
	int steps_per_sec = 64;  // how many steps do we want per seconds?
	bool set_dt_by_steps = true;  // choose wether we want to set dt by steps or by grid
	// dt will be set in cudaeuler, so that all changes can be applied there
	int snapshots_per_sec = 0;  // how many times do we want to save data per sec, set <= 0 to disable

	// set minor properties
	double incomp_threshhold = 1e-4;  // the maximum allowance of map to deviate from grad_chi begin 1
	double map_epsilon = 1e-5;  // distance used for foot points for GALS map advection

	// set memory properties
	int mem_RAM_GPU_remaps = 128;  // mem_index in MB on the GPU
	int mem_RAM_CPU_remaps = 4096;  // mem_RAM_CPU_remaps in MB on the CPU


	// set specific settings
	/*
	 * Time integration
	 * First order: "EulerExp"
	 * Second order: "AB2", "RK2"
	 * Third order: "RK3", "RK3Mod"
	 * Fourth order: "RK4", "RK4Mod"
	 */
	string time_integration = "RK3";

	// mapupdate order, "2nd", "4th", "6th"
	string map_update_order = "4th";

	// mollification settings, stencil size, 0, 4, 8
	int molly_stencil = 0;

	// for now we have two different upsample versions, upsample on vorticity and psi (1) or only psi (0)
	int upsample_version = 1;
	// in addition to the upsampling, we want to lowpass in fourier space by cutting high frequencies
	double freq_cut_psi = (double)(grid_coarse)/1.0;  // take into account, that frequencies are symmetric around N/2

	// skip remapping, usefull for convergence tests
	bool skip_remapping = false;


	// possibility to sample values on a specified grid
	bool sample_on_grid = false;
	int grid_sample = 2048;


	// set particles settings
	bool particles = false;  // en- or disable particles
	// tau_p has to be modified in cudaeuler, since it contains an array and i dont want to hardcode it here
	int particles_num = 1000;  // number of particles
	// Time integration for particles, define by name, "EulerExp", "Heun", "RK3", "RK4", "NicolasMid", "NicolasRK3"
	string particles_time_integration = "RK3";


	// make sure that not initialized values are set
	lagrange_order = 0;
	// now set everything
	setWorkspace(workspace); setSimName(sim_name);
	setGridCoarse(grid_coarse); setGridFine(grid_fine); setGridPsi(grid_psi);
	setFinalTime(final_time);
	setFactorDtByGrid(factor_dt_by_grid); setStepsPerSec(steps_per_sec);
	setSnapshotsPerSec(snapshots_per_sec); setSetDtBySteps(set_dt_by_steps);
	setInitialCondition(initial_condition);
	setIncompThreshold(incomp_threshhold);
	setMapEpsilon(map_epsilon);
	setMemRamGpuRemaps(mem_RAM_GPU_remaps); setMemRamCpuRemaps(mem_RAM_CPU_remaps);
	setTimeIntegration(time_integration);
	setMapUpdateOrder(map_update_order);
	setMollyStencil(molly_stencil);
	setUpsampleVersion(upsample_version);
	setFreqCutPsi(freq_cut_psi);
	setSkipRemapping(skip_remapping);
	setSampleOnGrid(sample_on_grid);
	setGridSample(grid_sample);
	setParticles(particles);
	if (particles) {
		setParticlesNum(particles_num);
		setParticlesTimeIntegration(particles_time_integration);
	}
}


/*
 *  Function to apply values taken from command line
 *  general form:  COMMAND=VALUE
 */
void SettingsCMM::applyCommands(int argc, char *args[]) {
	// loop over all commands
	for( int count = 0; count < argc; count++ ) {
		// construct string for command
		string command_full = args[count];
		// check for = sign
		int pos_equal = command_full.find("=");
		if (pos_equal != string::npos) {
			// construct two substrings
			string command = command_full.substr(0, pos_equal);
			string value = command_full.substr(pos_equal+1, command_full.length());

			// big if else for different commands
			if (command == "workspace") setWorkspace(value);
			else if (command == "sim_name") setSimName(value);
			else if (command == "grid_coarse") setGridCoarse(stoi(value));
			else if (command == "grid_fine") setGridFine(stoi(value));
			else if (command == "grid_psi") setGridPsi(stoi(value));
			else if (command == "final_time") setFinalTime(stod(value));
			else if (command == "factor_dt_by_grid") setFactorDtByGrid(stod(value));
			else if (command == "steps_per_sec") setStepsPerSec(stoi(value));
			else if (command == "set_dt_by_steps") {
				if (value == "true" || value == "True" || value == "1") setSetDtBySteps(true);
				else if (value == "false" || value == "False" || value == "0") setSetDtBySteps(false);
			}
			else if (command == "snapshots_per_sec") setSnapshotsPerSec(stoi(value));
			else if (command == "initial_condition") setInitialCondition(value);
			else if (command == "incomp_threshold") setIncompThreshold(stod(value));
			else if (command == "map_epsilon") setMapEpsilon(stod(value));
			else if (command == "mem_RAM_GPU_remaps") setMemRamGpuRemaps(stoi(value));
			else if (command == "mem_RAM_CPU_remaps") setMemRamCpuRemaps(stoi(value));
			else if (command == "time_integration") setTimeIntegration(value);
			else if (command == "map_update_order") setMapUpdateOrder(value);
			else if (command == "molly_stencil") setMollyStencil(stoi(value));
			else if (command == "upsample_version") setUpsampleVersion(stoi(value));
			else if (command == "freq_cut_psi") setFreqCutPsi(stod(value));
			else if (command == "skip_remapping") {
				if (value == "true" || value == "True" || value == "1") setSkipRemapping(true);
				else if (value == "false" || value == "False" || value == "0") setSkipRemapping(false);
			}
			else if (command == "sample_on_grid") {
				if (value == "true" || value == "True" || value == "1") setSampleOnGrid(true);
				else if (value == "false" || value == "False" || value == "0") setSampleOnGrid(false);
			}
			else if (command == "grid_sample") setGridSample(stoi(value));
			// true false handling
			else if (command == "particles") {
				if (value == "true" || value == "True" || value == "1") setParticles(true);
				else if (value == "false" || value == "False" || value == "0") setParticles(false);
			}
			else if (command == "particles_num") setParticlesNum(stoi(value));
			else if (command == "particles_time_integration") setParticlesTimeIntegration(value);
		}
	}
	//	 cout << "  args[" << count << "]   " << args[count] << "\n";
}


// class constructor from tfour main ingredients
SettingsCMM::SettingsCMM(string sim_name, int gridCoarse, int gridFine, string initialCondition) {
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
	// set presets
	setPresets();
}


// class constructor to take into account command line inputs
SettingsCMM::SettingsCMM(int argc, char *args[]) {
	// set presets
	setPresets();
	// override presets with command line arguments
	applyCommands(argc, args);
}
