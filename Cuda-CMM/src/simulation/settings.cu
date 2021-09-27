#include "settings.h"

void SettingsCMM::setPresets() {
	// grid settings for coarse and fine grid
	// 32		64		128		256		512		1024		2048		4096		8192		16384
	// max working on V100 : grid_scale = 4096; fine_grid_scale = 16384;
	int grid_coarse = 512;
	int grid_fine = 2048;
	int grid_psi = 1024;  // psi will be upsampled to this grid

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

	// set minor properties
	double incomp_threshhold = 1e-4;  // the maximum allowance of map to deviate from grad_chi begin 1
	double map_epsilon = 1e-4;  // distance used for foot points for GALS map advection

	// set memory properties
	int mem_RAM_GPU_remaps = 128;  // mem_index in MB on the GPU
	int mem_RAM_CPU_remaps = 4096;  // mem_RAM_CPU_remaps in MB on the CPU
	int Nb_array_RAM = 4;  // fixed for four different stacks

	// set specific settings
	// Time integration, define by name, "RKThree", "ABTwo", "EulerExp", "RKFour"
	string time_integration = "RKThree";

	// mapupdate order, "2nd", "3rd", "4th"
	string map_update_order = "3rd";

	// mollification settings, stencil size, 0, 4, 8
	int molly_stencil = 4;


	// now set everything
	setGridCoarse(grid_coarse);
	setGridFine(grid_fine);
	setGridPsi(grid_psi);
	setInitialCondition(initial_condition);
	setIncompThreshold(incomp_threshhold);
	setMapEpsilon(map_epsilon);
	setMemRamGpuRemaps(mem_RAM_GPU_remaps);
	setMemRamCpuRemaps(mem_RAM_CPU_remaps);
	setNbArrayRam(Nb_array_RAM);
	setTimeIntegration(time_integration);
	setMapUpdateOrder(map_update_order);
	setMollyStencil(molly_stencil);
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
			if (command == "grid_coarse") setGridCoarse(stoi(value));
			else if (command == "grid_fine") setGridFine(stoi(value));
			else if (command == "grid_psi") setGridPsi(stoi(value));
			else if (command == "initial_condition") setInitialCondition(value);
			else if (command == "incomp_threshold") setIncompThreshold(stoi(value));
			else if (command == "map_epsilon") setMapEpsilon(stoi(value));
			else if (command == "mem_RAM_GPU_remaps") setMemRamGpuRemaps(stoi(value));
			else if (command == "mem_RAM_CPU_remaps") setMemRamCpuRemaps(stoi(value));
			else if (command == "Nb_array_RAM") setNbArrayRam(stoi(value));
			else if (command == "time_integration") setTimeIntegration(value);
			else if (command == "map_update_order") setMapUpdateOrder(value);
			else if (command == "molly_stencil") setMollyStencil(stoi(value));
		}
	}
	//	 cout << "  args[" << count << "]   " << args[count] << "\n";
}


// class constructor from three main ingredients
SettingsCMM::SettingsCMM(int gridCoarse, int gridFine, string initialCondition) {
	// set presets
	setPresets();
	// override the three main components
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
