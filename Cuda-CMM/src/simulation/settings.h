// header file for settings
#ifndef __CMM_SETTINGS_H__
#define __CMM_SETTINGS_H__


#include <string>
#include <stdio.h>
#include <iostream>


using namespace std;


class SettingsCMM {

private:
	// main properties, needed to be able to run0
	string workspace, sim_name;
	int grid_coarse, grid_fine, grid_psi;
	string initial_condition; int initial_condition_num;
	// time stepping properties
	double final_time, factor_dt_by_grid;
	int steps_per_sec, snapshots_per_sec;
	bool set_dt_by_steps;
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
	int upsample_version;
	double freq_cut_psi;
	// particles
	bool particles;
	int particles_num;
	string particles_time_integration; int particles_time_integration_num;

public:
	// main functions
	SettingsCMM(string sim_name, int gridCoarse, int gridFine, string initialCondition);
	SettingsCMM();
	SettingsCMM(int argc, char *args[]);

	// function to set all values to preset ones
	void setPresets();
	// function to apply commands from command line
	void applyCommands(int argc, char *args[]);

	// get and set methods for all variables

	// name
	string getWorkspace() const { return workspace; }
	void setWorkspace(string Workspace) { workspace = Workspace; }
	string getSimName() const { return sim_name; }
	void setSimName(string simName) { sim_name = simName; }
	// grid settings
	int getGridCoarse() const { return grid_coarse; }
	void setGridCoarse(int gridCoarse) { grid_coarse = gridCoarse; }
	int getGridFine() const { return grid_fine; }
	void setGridFine(int gridFine) { grid_fine = gridFine; }
	int getGridPsi() const { return grid_psi; }
	void setGridPsi(int gridPsi) { grid_psi = gridPsi; }

	// time stepping properties
	double getFinalTime() const { return final_time; }
	void setFinalTime(double finalTime) { final_time = finalTime; }
	double getFactorDtByGrid() const { return factor_dt_by_grid; }
	void setFactorDtByGrid(double factorDtByGrid) { factor_dt_by_grid = factorDtByGrid; }
	int getStepsPerSec() const { return steps_per_sec; }
	void setStepsPerSec(int stepsPerSec) { steps_per_sec = stepsPerSec; }
	int getSnapshotsPerSec() const { return snapshots_per_sec; }
	void setSnapshotsPerSec(int snapshotsPerSec) { snapshots_per_sec = snapshotsPerSec; }
	bool getSetDtBySteps() const { return set_dt_by_steps; }
	void setSetDtBySteps(bool setDtBySteps) { set_dt_by_steps = setDtBySteps; }

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

	// upsample version, upsample vort and psi or only psi
	int getUpsampleVersion() const { return upsample_version; }
	void setUpsampleVersion(int upsampleVersion) { upsample_version = upsampleVersion; }

	// low pass cut frequencies for psi
	double getFreqCutPsi() const { return freq_cut_psi; }
	void setFreqCutPsi(double freqCutPsi) { freq_cut_psi = freqCutPsi; }

	// time integration for map advection
	string getTimeIntegration() const { return time_integration; }
	void setTimeIntegration(string timeIntegration) {
		time_integration = timeIntegration;
		if (timeIntegration == "EulerExp") time_integration_num = 0;
		else if (timeIntegration == "ABTwo") time_integration_num = 1;
		else if (timeIntegration == "RKThree") time_integration_num = 2;
		else if (timeIntegration == "RKFour") time_integration_num = 3;
		else if (timeIntegration == "RKThreeMod") time_integration_num = 4;
		else if (timeIntegration == "RKFourMod") time_integration_num = 5;
		else time_integration_num = -1;
	}
	int getTimeIntegrationNum() const { return time_integration_num; }

	// particles
	bool getParticles() const { return particles; }
	void setParticles(bool Particles) { particles = Particles; }
	int getParticlesNum() const { return particles_num; }
	void setParticlesNum(int particlesNum) { particles_num = particlesNum; }
	string getParticlesTimeIntegration() const { return particles_time_integration; }
	void setParticlesTimeIntegration(string particlesTimeIntegration) {
		particles_time_integration = particlesTimeIntegration;
		if (particlesTimeIntegration == "EulerExp") particles_time_integration_num = 0;
		else if (particlesTimeIntegration == "EulerMid") particles_time_integration_num = 1;
		else if (particlesTimeIntegration == "RKThree") particles_time_integration_num = 2;
		else if (particlesTimeIntegration == "NicolasMid") particles_time_integration_num = -2;
		else if (particlesTimeIntegration == "NicolasRKThree") particles_time_integration_num = -3;
		else particles_time_integration_num = -1;
	}
	int getParticlesTimeIntegrationNum() const { return particles_time_integration_num; }
};

#endif
