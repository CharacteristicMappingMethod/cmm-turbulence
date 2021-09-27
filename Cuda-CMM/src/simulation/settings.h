// header file for settings
#ifndef __CMM_SETTINGS_H__
#define __CMM_SETTINGS_H__


#include <string>
#include <stdio.h>
#include <iostream>


using namespace std;


class SettingsCMM {

private:
	// main properties, needed to be able to run
	int grid_coarse, grid_fine, grid_psi;
	string initial_condition; int initial_condition_num;
	// minor properties, can be tweaked but mainly remain constant
	double incomp_threshhold;
	double map_epsilon;
	//memory variables
	int mem_RAM_GPU_remaps;
	int mem_RAM_CPU_remaps;
	int Nb_array_RAM;
	// specific
	string time_integration; int time_integration_num;
	string map_update_order; int map_update_order_num;
	int molly_stencil;

public:
	// main functions
	SettingsCMM(int gridCoarse, int gridFine, string initialCondition);
	SettingsCMM();
	SettingsCMM(int argc, char *args[]);

	// function to set all values to preset ones
	void setPresets();
	// function to apply commands from command line
	void applyCommands(int argc, char *args[]);

	// get and set methods for all variables

	// grid settings
	int getGridCoarse() const { return grid_coarse; }
	void setGridCoarse(int gridCoarse) { grid_coarse = gridCoarse; }
	int getGridFine() const { return grid_fine; }
	void setGridFine(int gridFine) { grid_fine = gridFine; }
	int getGridPsi() const { return grid_psi; }
	void setGridPsi(int gridPsi) { grid_psi = gridPsi; }

	// initial conditions setting
	string getInitialCondition() const { return initial_condition; }
	void setInitialCondition(string initialCondition) {
		initial_condition = initialCondition;
		// tied to num, for faster handling
		if(initialCondition == "4_nodes") initial_condition_num = 0;
		else if(initialCondition == "quadropole") initial_condition_num = 1;
		else if(initialCondition == "two_votices") initial_condition_num = 2;
		else if(initialCondition == "three_vortices") initial_condition_num = 3;
		else if(initialCondition == "single_shear_layer") initial_condition_num = 4;
		else if(initialCondition == "turbulence_gaussienne") initial_condition_num = 5;
		else initial_condition_num = -1;
	}
	int getInitialConditionNum() const { return initial_condition_num; }

	// minor properties
	double getIncompThreshold() const { return incomp_threshhold; }
	void setIncompThreshold(double incompThreshhold) { incomp_threshhold = incompThreshhold; }
	double getMapEpsilon() const { return map_epsilon; }
	void setMapEpsilon(double mapEpsilon) { map_epsilon = mapEpsilon; }

	// memory variables
	int getMemRamGpuRemaps() const { return mem_RAM_GPU_remaps; }
	void setMemRamGpuRemaps(int memRamGpuRemaps) { mem_RAM_GPU_remaps = memRamGpuRemaps; }
	int getMemRamCpuRemaps() const { return mem_RAM_CPU_remaps; }
	void setMemRamCpuRemaps(int memRamCpuRemaps) { mem_RAM_CPU_remaps = memRamCpuRemaps; }
	int getNbArrayRam() const { return Nb_array_RAM; }
	void setNbArrayRam(int nbArrayRam) { Nb_array_RAM = nbArrayRam; }

	// map update order handling the stencil of footpoints
	string getMapUpdateOrder() const { return map_update_order; }
	void setMapUpdateOrder(string mapUpdateOrder) {
		map_update_order = mapUpdateOrder;
		// tied to num, for faster handling
		if (mapUpdateOrder == "2nd") map_update_order_num = 0;
		else if (mapUpdateOrder == "3rd") map_update_order_num = 1;
		else if (mapUpdateOrder == "4th") map_update_order_num = 2;
		else map_update_order_num = -1;
	}
	int getMapUpdateOrderNum() const { return map_update_order_num; }

	// molly stencil, amount of points to be used for mollification
	int getMollyStencil() const { return molly_stencil; }
	void setMollyStencil(int mollyStencil) { molly_stencil = mollyStencil; }

	// time integration for map advection
	string getTimeIntegration() const { return time_integration; }
	void setTimeIntegration(string timeIntegration) {
		time_integration = timeIntegration;
		if (timeIntegration == "EulerExp") time_integration_num = 0;
		else if (timeIntegration == "ABTwo") time_integration_num = 1;
		else if (timeIntegration == "RKThree") time_integration_num = 2;
		else if (timeIntegration == "RKFour") time_integration_num = 3;
		else time_integration_num = -1;
	}
	int getTimeIntegrationNum() const { return time_integration_num; }
};

#endif
