#include "settings.h"

void SettingsCMM::setPresets() {
	// naming and saving settings of the simulation
	string workspace = "./"; // where should the files be saved? "./" or "" means at the run location, has to end with backslash
	string sim_name = "debug";  // unique identifier to differentiate simulations

	// grid settings for coarse and fine grid
	// 32		64		128		256		512		1024		2048		4096		8192		16384
	// max working on V100 : grid_scale = 4096; fine_grid_scale = 16384;
	int grid_coarse = 512;
	int grid_fine = 2048;
	int grid_psi = 1024;  // psi will be (up)sampled on this grid
	int grid_vort = grid_psi;  // vorticity will be sampled on this grid for computation of psi, this changes the scales of the vorticity

	/*
	 *  Initial conditions
	 *  "4_nodes" 				-	flow containing exactly 4 fourier modes with two vortices
	 *  "quadropole"			-	???
	 *  "three_vortices"		-	???
	 *  "single_shear_layer"	-	shear layer problem forming helmholtz-instabilities, merging into two vortices which then merges into one big vortex
	 *  "two_vortices"			-	???
	 *  "turbulence_gaussienne"	-	???
	 *  "shielded_vortex"		-	vortex core with ring of negative vorticity around it
	 */
	string initial_condition = "4_nodes";
//	string initial_condition = "shielded_vortex";

	// set time properties
	double final_time = 1;  // end of computation
	double factor_dt_by_grid = 1;  // if dt is set by the grid (cfl), then this should be the max velocity
	int steps_per_sec = 64;  // how many steps do we want per seconds?
	bool set_dt_by_steps = true;  // choose whether we want to set dt by steps or by grid
	// dt will be set in cudaeuler, so that all changes can be applied there
	int snapshots_per_sec = 1;  // how many times do we want to save data per sec, set <= 0 to disable
	bool save_initial = true;  // consume less data and make it possible to disable saving the initial data
	bool save_final = true;  // consume less data and make it possible to disable saving the final data

	// set minor properties
	double incomp_threshhold = 1e-4;  // the maximum allowance of map to deviate from grad_chi begin 1
	double map_epsilon = 1e-3;  // distance used for foot points for GALS map advection
//	double map_epsilon = 6.283185307179/512.0;  // distance used for foot points for GALS map advection
	// skip remapping, usefull for convergence tests
	bool skip_remapping = false;

	// set memory properties
	int mem_RAM_CPU_remaps = 4096;  // mem_RAM_CPU_remaps in MB on the CPU
	bool save_map_stack = false;  // possibility to save the map stack to reuse for other computations to skip initial time

	// set specific settings
	/*
	 * Time integration
	 * First order: "EulerExp"
	 * Second order: "AB2", "RK2"
	 * Third order: "RK3", "RK3Mod"
	 * Fourth order: "RK4", "RK4Mod"
	 */
	string time_integration = "RK3Mod";

	// mapupdate order, "2nd", "4th", "6th"
	string map_update_order = "4th";
	bool map_update_grid = false;  // should map update be computed with grid or footpoints?

	// mollification settings, stencil size, 0, 4, 8
	int molly_stencil = 0;

	// in addition to the upsampling, we want to lowpass in fourier space by cutting high frequencies
	double freq_cut_psi = (double)(grid_coarse)/2.0;  // take into account, that frequencies are symmetric around N/2

	// possibility to sample values on a specified grid
	bool sample_on_grid = false;
	int grid_sample = 1024;
	int sample_snapshots_per_sec = snapshots_per_sec;  // how many times do we want to save sample data per sec, set <= 0 to disable
	bool sample_save_initial = true;  // consume less data and make it possible to disable saving the initial data
	bool sample_save_final = true;  // consume less data and make it possible to disable saving the final data


	// set particles settings
	bool particles = false;  // en- or disable particles
	int particles_num = 1000;  // number of particles

	int particles_tau_num = 3;  // how many tau_p values do we have? for now maximum is 100
//	double Tau_p[Nb_Tau_p] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.25, 0.5, 0.75, 1, 2, 5, 13};
	string particles_tau_s = """0, 0.01, 1""";  // all those "s are needed, to have one pair of "s in the string for the conversion

	int particles_snapshots_per_sec = snapshots_per_sec;  // how many times do we want to save particles per sec, set <= 0 to disable
	bool particles_save_initial = true;  // consume less data and make it possible to disable saving the initial data
	bool particles_save_final = true;  // consume less data and make it possible to disable saving the final data
	int particles_steps = -1;  // hackery for particle convergence

	bool save_fine_particles = false;  // wether or not we want to save fine particles
	int particles_fine_num = 1000;  // number of particles where every time step the position will be saved
	/*
	 * Time integration for particles
	 * First order: "EulerExp"
	 * Second order: "AB2", "RK2"
	 * Third order: "RK3", "RK3Mod"
	 * Fourth order: "RK4", "RK4Mod"
	 * Nicolas: "NicolasMid", "NicolasRK3"
	 */
	string particles_time_integration = "RK3";


	// make sure that not initialized values are set
	lagrange_order = 0;
//	lagrange_order = 4;  // rk tests
	// now set everything
	setWorkspace(workspace); setSimName(sim_name);
	setGridCoarse(grid_coarse); setGridFine(grid_fine);
	setGridPsi(grid_psi); setGridVort(grid_vort);
	setFinalTime(final_time);
	setFactorDtByGrid(factor_dt_by_grid); setStepsPerSec(steps_per_sec);
	setSnapshotsPerSec(snapshots_per_sec); setSetDtBySteps(set_dt_by_steps);
	setSaveInitial(save_initial); setSaveFinal(save_final);
	setInitialCondition(initial_condition);

	setMemRamCpuRemaps(mem_RAM_CPU_remaps);
	setSaveMapStack(save_map_stack);

	setIncompThreshold(incomp_threshhold);
	setMapEpsilon(map_epsilon);
	setTimeIntegration(time_integration);
	setMapUpdateOrder(map_update_order);
	setMapUpdateGrid(map_update_grid);
	setMollyStencil(molly_stencil);
	setFreqCutPsi(freq_cut_psi);
	setSkipRemapping(skip_remapping);

	setSampleOnGrid(sample_on_grid);
	setGridSample(grid_sample);
	setSampleSnapshotsPerSec(sample_snapshots_per_sec);
	setSampleSaveInitial(sample_save_initial); setSampleSaveFinal(sample_save_final);

	setParticles(particles);
	setParticlesNum(particles_num);
	setParticlesTauNum(particles_tau_num);
	string_to_double_array(particles_tau_s, particles_tau);

	setParticlesSnapshotsPerSec(particles_snapshots_per_sec);
	setParticlesSaveInitial(particles_save_initial); setParticlesSaveFinal(particles_save_final);

	setSaveFineParticles(save_fine_particles);
	setParticlesFineNum(particles_fine_num);

	setParticlesTimeIntegration(particles_time_integration);

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
		string command_full = args[count];
		// check for = sign
		int pos_equal = command_full.find("=");
		if (pos_equal != string::npos) {
			// construct two substrings
			string command = command_full.substr(0, pos_equal);
			string value = command_full.substr(pos_equal+1, command_full.length());

			// big if else for different commands
			// this beast is becoming larger and larger, i should convert it to something more automatic
			// link to site for that: https://stackoverflow.com/questions/4480788/c-c-switch-case-with-string
			if (command == "workspace") setWorkspace(value);
			else if (command == "sim_name") setSimName(value);
			else if (command == "grid_coarse") setGridCoarse(stoi(value));
			else if (command == "grid_fine") setGridFine(stoi(value));
			else if (command == "grid_psi") setGridPsi(stoi(value));
			else if (command == "grid_vort") setGridVort(stoi(value));

			else if (command == "final_time") setFinalTime(stod(value));
			else if (command == "factor_dt_by_grid") setFactorDtByGrid(stod(value));
			else if (command == "steps_per_sec") setStepsPerSec(stoi(value));
			else if (command == "set_dt_by_steps") setSetDtBySteps(getBoolFromString(value));
			else if (command == "save_initial") setSaveInitial(getBoolFromString(value));
			else if (command == "save_final") setSaveFinal(getBoolFromString(value));
			else if (command == "snapshots_per_sec") setSnapshotsPerSec(stoi(value));

			else if (command == "mem_RAM_CPU_remaps") setMemRamCpuRemaps(stoi(value));
			else if (command == "save_map_stack") setSaveMapStack(getBoolFromString(value));

			else if (command == "initial_condition") setInitialCondition(value);
			else if (command == "incomp_threshold") setIncompThreshold(stod(value));
			else if (command == "map_epsilon") setMapEpsilon(stod(value));
			else if (command == "time_integration") setTimeIntegration(value);
			else if (command == "map_update_order") setMapUpdateOrder(value);
			else if (command == "map_update_order") setMapUpdateGrid(getBoolFromString(value));
			else if (command == "molly_stencil") setMollyStencil(stoi(value));
			else if (command == "freq_cut_psi") setFreqCutPsi(stod(value));
			else if (command == "skip_remapping") setSkipRemapping(getBoolFromString(value));

			else if (command == "sample_on_grid") setSampleOnGrid(getBoolFromString(value));
			else if (command == "grid_sample") setGridSample(stoi(value));
			else if (command == "sample_save_initial") setSampleSaveInitial(getBoolFromString(value));
			else if (command == "sample_save_final") setSampleSaveFinal(getBoolFromString(value));
			else if (command == "sample_snapshots_per_sec") setSampleSnapshotsPerSec(stoi(value));

			else if (command == "particles") setParticles(getBoolFromString(value));
			else if (command == "particles_num") setParticlesNum(stoi(value));
			else if (command == "particles_tau_num") setParticlesTauNum(stoi(value));
			else if (command == "particles_tau") string_to_double_array(value, particles_tau);
			else if (command == "particles_snapshots_per_sec") setParticlesSnapshotsPerSec(stoi(value));
			else if (command == "particles_save_initial") setParticlesSaveInitial(getBoolFromString(value));
			else if (command == "particles_save_final") setParticlesSaveFinal(getBoolFromString(value));
			else if (command == "save_fine_particles") setSaveFineParticles(getBoolFromString(value));
			else if (command == "particles_fine_num") setParticlesFineNum(stoi(value));
			else if (command == "particles_time_integration") setParticlesTimeIntegration(value);

			// hackery for particle convergence
			else if (command == "particles_steps") setParticlesSteps(stoi(value));
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


// little helper function to combine parsing the bool value
bool SettingsCMM::getBoolFromString(string value) {
	if (value == "true" || value == "True" || value == "1") return true;
	else if (value == "false" || value == "False" || value == "0") return false;
	return false;  // in case no value is chosen
}


// little helper functions to be able to give in arrays
void SettingsCMM::string_to_double_array(string s_array, double *array) {
	// Attention : no parse check implemented, this is not good habit, but i didn't come up with a nice working unique thing
	// erase " if surrounded by it
	if (s_array.substr(0, 1) == """" || s_array.substr(s_array.length()-1, s_array.length()) == """") {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
	}
	// loop over all elements
	int index = 0;
	std::string::size_type pos = 0; std::string::size_type pos_new = 0;
	do {
		pos_new = s_array.find(",", pos );
		string substring;
		if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new);
		else substring = s_array.substr(pos, s_array.length());
		array[index] = stod(substring);
		index++;
		pos = pos_new + 1;
	} while (pos_new != std::string::npos);
}
void SettingsCMM::string_to_int_array(string s_array, int *array) {
	// Attention : no parse check implemented, this is not good habit, but i didn't come up with a nice working unique thing
	// erase " if surrounded by it
	if (s_array.substr(0, 1) == """" || s_array.substr(s_array.length()-1, s_array.length()) == """") {
		s_array.erase(0, 1); s_array.erase(s_array.length()-1, s_array.length());  // erase brackets
	}
	// loop over all elements
	int index = 0;
	std::string::size_type pos = 0; std::string::size_type pos_new = 0;
	do {
		pos_new = s_array.find(",", pos );
		string substring;
		if (pos_new != std::string::npos) substring = s_array.substr(pos, pos_new);
		else substring = s_array.substr(pos, s_array.length());
		array[index] = stoi(substring);
		index++;
		pos = pos_new + 1;
	} while (pos_new != std::string::npos);
}
