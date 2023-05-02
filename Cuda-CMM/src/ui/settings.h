/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/CharacteristicMappingMethod/cmm-turbulence
*
******************************************************************************************************************************/

// header file for settings
#ifndef __CMM_SETTINGS_H__
#define __CMM_SETTINGS_H__


#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include "globals.h"

#define T_MAX 10000  // maximum time, used to simplify saving and stuff


template <typename T>
bool parseString(std::string str, T* arr) {
	// Check if the string is empty or does not start and end with curly braces
	if (str.empty() || (str[0] != '{' && str[0] != '"') || (str[str.length() - 1] != '}' && str[str.length() - 1] != '"')) {
		return false;
	}

	// Remove the curly braces from the string
	str.erase(0, 1); str.erase(str.length() - 1, 1);

	// Create a stringstream from the modified string
	std::stringstream ss(str);
	int counter = 0;  // Initialize the size of the array

	// Extract the values from the stringstream and insert them into the array
	T val;
	while (ss >> val) {
		arr[counter++] = val;
		if (ss.peek() == ',') {
			ss.ignore();
		}
	}
	return true;
}
template <typename T>
std::string arrayToString(T* arr, int size) {
	std::stringstream ss;

	// Insert the values from the array into the stringstream
	ss << "{";
	for (int i = 0; i < size; i++) {
		ss << arr[i];
		if (i < size - 1) {
			ss << ",";
		}
	}
	ss << "}";

	// Return the string from the stringstream
	return ss.str();
}

// class to define the saving of computational variables
class SaveComputational {

public:
	// attributes, not modern style but i was too lazy to include getters and setters
	bool is_instant;
	double time_start, time_end, time_step;
	std::string var;
	bool conv;

	void setAllVariables(std::string value);
	void setVariable(std::string value);
	std::string getVariables();
};

// class to define the saving of sample variables
class SaveSample {

public:
	// attributes, not modern style but i was too lazy to include getters and setters
	bool is_instant;
	double time_start, time_end, time_step;
	std::string var;
	int grid;

	void setAllVariables(std::string value);
	void setVariable(std::string value);
	std::string getVariables();
};

// class to define the saving of zooms
class SaveZoom {

public:
	// attributes, not modern style but i was too lazy to include getters and setters
	bool is_instant;
	double time_start, time_end, time_step;
	std::string var;
	int grid;
	double pos_x, pos_y, width_x, width_y;
	int rep; double rep_fac;

	void setAllVariables(std::string value);
	void setVariable(std::string value);
	std::string getVariables();
};


// class to define the computations for particles being advected
class ParticlesAdvected {

public:
	// attributes, not modern style but i was too lazy to include getters and setters
	int num;
	double tau;
	unsigned long long seed;
	std::string time_integration; int time_integration_num;
	std::string init_name; int init_num; bool init_vel;
	double init_time, init_param_1, init_param_2, init_param_3, init_param_4;

	void setAllVariables(std::string value);
	void setVariable(std::string value);
	std::string getVariables();
};


// class to define the computations for particles being forwarded
class ParticlesForwarded {

public:
	// attributes, not modern style but i was too lazy to include getters and setters
	int num;
	unsigned long long seed;
	std::string init_name; int init_num;
	double init_time; int init_map;  // map needed to know from which map we advect
	double init_param_1, init_param_2, init_param_3, init_param_4;

	void setAllVariables(std::string value);
	void setVariable(std::string value);
	std::string getVariables();
};


class SettingsCMM {

private:
	// main properties, needed to be able to run
	std::string workspace, sim_name, file_name, simulation_type;
	int grid_coarse, grid_fine, grid_psi, grid_vort;
	std::string initial_condition; double initial_params[10] = {0};
	std::string initial_discrete_location;
	int initial_condition_num, initial_discrete_grid; bool initial_discrete;
	int verbose;
	// time stepping properties
	double final_time, factor_dt_by_grid;
	int steps_per_sec;
	bool set_dt_by_steps;
	// save computational
	int save_computational_num;
	SaveComputational* save_computational;
	// minor properties, can be tweaked but mainly remain constant
	double incomp_threshhold;
	double map_epsilon;
	//memory variables
	int mem_RAM_CPU_remaps; bool save_map_stack;
	double restart_time; std::string restart_location;
	// specific
	std::string time_integration; int time_integration_num;
	int lagrange_order, lagrange_override;
	bool lagrange_init_higher_order;
	std::string map_update_order; int map_update_order_num;
	int molly_stencil;
	double freq_cut_psi;
	bool skip_remapping;
	// sample grid
	int save_sample_num;
	SaveSample* save_sample;
	// passive scalar
	std::string scalar_name, scalar_discrete_location;
	int scalar_num, scalar_discrete_grid; bool scalar_discrete;
	// zoom
	int save_zoom_num;
	SaveZoom* save_zoom;
	// forward map
	bool forward_map;
	int particles_forwarded_num;
	ParticlesForwarded* particles_forwarded;
	// particles
	int particles_advected_num;
	ParticlesAdvected* particles_advected;

	int particles_steps;

public:
	// arrays are hard to deal with methods in c++, so i make them public, 100 is taken just to have enough space
	double forward_particles_init_parameter[100];

	// main functions
	SettingsCMM(std::string sim_name, int gridCoarse, int gridFine, std::string initialCondition);
	SettingsCMM();
	SettingsCMM(int argc, char *args[]);

	// function to set all values to preset ones
	void setPresets();
	// function to apply commands from command line
	void applyCommands(int argc, char *args[]);
	// function to set variables, useful for param file or command line args
	int setVariable(std::string command_full, std::string delimiter);
	void setSaveComputational(std::string command_full, std::string delimiter, int number);
	void setSaveSample(std::string command_full, std::string delimiter, int number);
	void setSaveZoom(std::string command_full, std::string delimiter, int number);
	void setParticlesAdvected(std::string command_full, std::string delimiter, int number);
	void setParticlesForwarded(std::string command_full, std::string delimiter, int number);


	// get and set methods for all variables, this is quite much and quite ugly, i know

	// name
	std::string getWorkspace() const { return workspace; }
	void setWorkspace(std::string Workspace) { workspace = Workspace; }
	std::string getSimulationType() const { return simulation_type; }
	void setSimulationType(std::string SimulationType) { simulation_type = SimulationType; }
	std::string getSimName() const { return sim_name; }
	void setSimName(std::string simName) { sim_name = simName; }
	std::string getFileName() const { return file_name; }
	void setFileName(std::string fileName) { file_name = fileName; }
	// grid settings
	int getGridCoarse() const { return grid_coarse; }
	void setGridCoarse(int gridCoarse) { grid_coarse = gridCoarse; }
	int getGridFine() const { return grid_fine; }
	void setGridFine(int gridFine) { grid_fine = gridFine; }
	int getGridPsi() const { return grid_psi; }
	void setGridPsi(int gridPsi) { grid_psi = gridPsi; }
	int getGridVort() const { return grid_vort; }
	void setGridVort(int gridVort) { grid_vort = gridVort; }


	// time stepping properties
	double getFinalTime() const { return final_time; }
	void setFinalTime(double finalTime) { final_time = finalTime; }
	double getFactorDtByGrid() const { return factor_dt_by_grid; }
	void setFactorDtByGrid(double factorDtByGrid) { factor_dt_by_grid = factorDtByGrid; }
	int getStepsPerSec() const { return steps_per_sec; }
	void setStepsPerSec(int stepsPerSec) { steps_per_sec = stepsPerSec; }
//	double getSnapshotsPerSec() const { return snapshots_per_sec; }
//	void setSnapshotsPerSec(double snapshotsPerSec) { snapshots_per_sec = snapshotsPerSec; }
	bool getSetDtBySteps() const { return set_dt_by_steps; }
	void setSetDtBySteps(bool setDtBySteps) { set_dt_by_steps = setDtBySteps; }

	int getSaveComputationalNum() const { return save_computational_num; }
	void setSaveComputationalNum(int saveComputationalNum) {
		save_computational_num = saveComputationalNum;
		delete [] save_computational;
		save_computational = new SaveComputational[save_computational_num];
	}
	SaveComputational* getSaveComputational() const { return save_computational; }


	// initial conditions setting
	std::string getInitialCondition() const { return initial_condition; }
	void setInitialCondition(std::string initialCondition) {
		initial_condition = initialCondition;
		// tied to num, for faster handling
		if (simulation_type=="cmm_vlasov_poisson_1d"){
			if(initialCondition == "landau_damping") initial_condition_num = 0;
			else if(initialCondition == "two_stream") initial_condition_num = 1;
			else {
				initial_condition = "zero";  // make clear, that setting the initial condition failed
				initial_condition_num = -1;
				error("Ok, try it again please with an existing initial condition.! Currently: " + initialCondition, 230304);
				// throw runtime_error("test");
			}
		}
		else
		{
			if(initialCondition == "4_nodes") initial_condition_num = 0;
			else if(initialCondition == "quadropole") initial_condition_num = 1;
			else if(initialCondition == "one_vortex") initial_condition_num = 2;
			else if(initialCondition == "two_votices") initial_condition_num = 3;
			else if(initialCondition == "three_vortices") initial_condition_num = 4;
			else if(initialCondition == "single_shear_layer") initial_condition_num = 5;
			else if(initialCondition == "tanh_shear_layer") initial_condition_num = 6;
			else if(initialCondition == "turbulence_gaussienne") initial_condition_num = 7;
			else if(initialCondition == "gaussian_blobs") initial_condition_num = 8;
			else if(initialCondition == "shielded_vortex") initial_condition_num = 9;
			else if(initialCondition == "two_cosine") initial_condition_num = 10;
			else if(initialCondition == "vortex_sheets") initial_condition_num = 11;
			else {
				initial_condition = "zero";  // make clear, that setting the initial condition failed
				initial_condition_num = -1;
				error("Ok, try it again please with an existing initial condition.! Currently: " + initialCondition,230305);

			}
		}
	}
	int getInitialConditionNum() const { return initial_condition_num; }
	void setInitialParams(std::string initialParams) { parseString(initialParams, initial_params);}
	std::string getInitialParams() const { return arrayToString(initial_params, 10); }
	double* getInitialParamsPointer() { return initial_params; }
	bool getInitialDiscrete() const { return initial_discrete; }
	void setInitialDiscrete(bool initialDiscrete) { initial_discrete = initialDiscrete; }
	int getInitialDiscreteGrid() const { return initial_discrete_grid; }
	void setInitialDiscreteGrid(int initialDiscreteGrid) { initial_discrete_grid = initialDiscreteGrid; }
	std::string getInitialDiscreteLocation() const { return initial_discrete_location; }
	void setInitialDiscreteLocation(std::string initialDiscreteLocation) { initial_discrete_location = initialDiscreteLocation; }

	int getVerbose() const { return verbose; }
	void setVerbose(int Verbose) { verbose = Verbose; }


	// minor properties
	double getIncompThreshold() const { return incomp_threshhold; }
	void setIncompThreshold(double incompThreshhold) { incomp_threshhold = incompThreshhold; }
	double getMapEpsilon() const { return map_epsilon; }
	void setMapEpsilon(double mapEpsilon) { map_epsilon = mapEpsilon; }

	// memory variables
	int getMemRamCpuRemaps() const { return mem_RAM_CPU_remaps; }
	void setMemRamCpuRemaps(int memRamCpuRemaps) { mem_RAM_CPU_remaps = memRamCpuRemaps; }
	bool getSaveMapStack() const { return save_map_stack; }
	void setSaveMapStack(bool saveMapStack) { save_map_stack = saveMapStack; }
	double getRestartTime() const { return restart_time; }
	void setRestartTime(double restartTime) { restart_time = restartTime; }
	std::string getRestartLocation() const { return restart_location; }
	void setRestartLocation(std::string restartLocation) { restart_location = restartLocation; }

	// map update order handling the stencil of footpoints
	std::string getMapUpdateOrder() const { return map_update_order; }
	void setMapUpdateOrder(std::string mapUpdateOrder) {
		map_update_order = mapUpdateOrder;
		// tied to num, for faster handling
		if (mapUpdateOrder == "2nd") map_update_order_num = 0;
		else if (mapUpdateOrder == "4th") map_update_order_num = 1;
		else if (mapUpdateOrder == "6th") map_update_order_num = 2;
		else map_update_order_num = -1;
	}
	int getMapUpdateOrderNum() const { return map_update_order_num; }

	// molly stencil, amount of points to be used for mollification
	int getMollyStencil() const { return molly_stencil; }
	void setMollyStencil(int mollyStencil) { molly_stencil = mollyStencil; }

	// low pass cut frequencies for psi
	double getFreqCutPsi() const { return freq_cut_psi; }
	void setFreqCutPsi(double freqCutPsi) { freq_cut_psi = freqCutPsi; }

	// time integration for map advection
	std::string getTimeIntegration() const { return time_integration; }
	void setTimeIntegration(std::string timeIntegration) {
		time_integration = timeIntegration;
		if (timeIntegration == "EulerExp") { time_integration_num = 10; if (getLagrangeOrder() < 1) lagrange_order = 1; }
		else if (timeIntegration == "RK1") { time_integration_num = 10; if (getLagrangeOrder() < 1) lagrange_order = 1; }
		else if (timeIntegration == "Heun") { time_integration_num = 21; if (getLagrangeOrder() < 2) lagrange_order = 2; }
		else if (timeIntegration == "RK2") { time_integration_num = 21; if (getLagrangeOrder() < 2) lagrange_order = 2; }
		else if (timeIntegration == "AB2") { time_integration_num = 20; if (getLagrangeOrder() < 2) lagrange_order = 2; }
		else if (timeIntegration == "RK3") { time_integration_num = 30; if (getLagrangeOrder() < 3) lagrange_order = 3; }
		else if (timeIntegration == "RK4") { time_integration_num = 40; if (getLagrangeOrder() < 4) lagrange_order = 4; }
		else if (timeIntegration == "RK3Mod") { time_integration_num = 31; if (getLagrangeOrder() < 3) lagrange_order = 3; }
		else if (timeIntegration == "RK4Mod") { time_integration_num = 41; if (getLagrangeOrder() < 4) lagrange_order = 4; }
		else time_integration_num = -1;
	}
	int getTimeIntegrationNum() const { return time_integration_num; }

	// lagrange order set indirectly from map and particle time integration
	int getLagrangeOrder() const { return lagrange_order; }
	void setLagrangeOrder(int lagrangeOrder) { lagrange_order = lagrangeOrder; }
	// lagrange override to force a specific lagrange order
	int getLagrangeOverride() const { return lagrange_override; }
	void setLagrangeOverride(int lagrangeOverride) { lagrange_override = lagrangeOverride; }
	// lagrange init setting
	bool getLagrangeInitHigherOrder() const { return lagrange_init_higher_order; }
	void setLagrangeInitHigherOrder(bool lagrangeInitHigherOrder) { lagrange_init_higher_order = lagrangeInitHigherOrder; }

	// skip remapping
	bool getSkipRemapping() const { return skip_remapping; }
	void setSkipRemapping(bool skipRemapping) { skip_remapping = skipRemapping; }


	// possibility to sample on specific grid
	int getSaveSampleNum() const { return save_sample_num; }
	void setSaveSampleNum(int saveSampleNum) {
		save_sample_num = saveSampleNum;
		delete [] save_sample;
		save_sample = new SaveSample[save_sample_num];
	}
	SaveSample* getSaveSample() const { return save_sample; }


	// passive scalar
	std::string getScalarName() const { return scalar_name; }
	void setScalarName(std::string scalarName) {
		scalar_name = scalarName;
		// tied to num, for faster handling
		if(scalarName == "rectangle") scalar_num = 0;
		else if(scalarName == "gaussian") scalar_num = 1;
		else if(scalarName == "circular_ring") scalar_num = 2;
		else scalar_num = -1;
	}
	int getScalarNum() const { return scalar_num; }
	bool getScalarDiscrete() const { return scalar_discrete; }
	void setScalarDiscrete(bool scalarDiscrete) { scalar_discrete = scalarDiscrete; }
	int getScalarDiscreteGrid() const { return scalar_discrete_grid; }
	void setScalarDiscreteGrid(int scalarDiscreteGrid) { scalar_discrete_grid = scalarDiscreteGrid; }
	std::string getScalarDiscreteLocation() const { return scalar_discrete_location; }
	void setScalarDiscreteLocation(std::string scalarDiscreteLocation) { scalar_discrete_location = scalarDiscreteLocation; }



	// zoom
	int getSaveZoomNum() const { return save_zoom_num; }
	void setSaveZoomNum(int saveZoomNum) {
		save_zoom_num = saveZoomNum;
		delete [] save_zoom;
		save_zoom = new SaveZoom[save_zoom_num];
	}
	SaveZoom* getSaveZoom() const { return save_zoom; }


	// forward map
	bool getForwardMap() const { return forward_map; }
	void setForwardMap(bool forwardMap) { forward_map = forwardMap; }

	int getParticlesForwardedNum() const { return particles_forwarded_num; }
	void setParticlesForwardedNum(int particlesForwardedNum) {
		particles_forwarded_num = particlesForwardedNum;
		delete [] particles_forwarded;
		particles_forwarded = new ParticlesForwarded[particles_forwarded_num];
	}
	ParticlesForwarded* getParticlesForwarded() const { return particles_forwarded; }


	// particles
	int getParticlesAdvectedNum() const { return particles_advected_num; }
	void setParticlesAdvectedNum(int particlesAdvectedNum) {
		particles_advected_num = particlesAdvectedNum;
		delete [] particles_advected;
		particles_advected = new ParticlesAdvected[particles_advected_num];
	}
	ParticlesAdvected* getParticlesAdvected() const { return particles_advected; }

	int getParticlesSteps() const { return particles_steps; }
	void setParticlesSteps(int particlesSteps) { particles_steps = particlesSteps; }
};


// format datatype to string with high precision, needed to have variables in string arrays
template<typename Type> std::string str_t (const Type & t)
{
  std::ostringstream os;
  os.precision(16);
  os << t;
  return os.str ();
}


// little helper function to combine parsing the bool value
bool getBoolFromString(std::string value);

#endif
