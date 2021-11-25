// header file for settings
#ifndef __CMM_SETTINGS_H__
#define __CMM_SETTINGS_H__


#include <string>
#include <stdio.h>
#include <iostream>


class SettingsCMM {

private:
	// main properties, needed to be able to run
	std::string workspace, sim_name, file_name;
	int grid_coarse, grid_fine, grid_psi, grid_vort;
	std::string initial_condition; int initial_condition_num;
	int verbose;
	// time stepping properties
	double final_time, factor_dt_by_grid, snapshots_per_sec;
	int steps_per_sec;
	bool set_dt_by_steps;
	bool save_initial, save_final;
	// minor properties, can be tweaked but mainly remain constant
	double incomp_threshhold;
	double map_epsilon;
	//memory variables
	int mem_RAM_CPU_remaps; bool save_map_stack;
	// specific
	std::string time_integration; int time_integration_num;
	int lagrange_order, lagrange_override;
	std::string map_update_order; int map_update_order_num;
	bool map_update_grid;
	int molly_stencil;
	double freq_cut_psi;
	bool skip_remapping;
	// sample grid
	bool sample_on_grid;
	int grid_sample;
	double sample_snapshots_per_sec;
	bool sample_save_initial, sample_save_final, conv_init_final;
	// zoom
	bool zoom, zoom_save_psi, zoom_save_particles;
	double zoom_center_x, zoom_center_y, zoom_width_x, zoom_width_y, zoom_repetitions_factor, zoom_snapshots_per_sec;
	int grid_zoom, zoom_repetitions;
	bool zoom_save_initial, zoom_save_final;
	// particles
	bool particles, save_fine_particles;
	unsigned long long particles_seed;
	double particles_center_x, particles_center_y, particles_width_x, particles_width_y;
	int particles_tau_num;
	bool particles_save_initial, particles_save_final;
	int particles_num, particles_fine_num;
	double particles_snapshots_per_sec;
	std::string particles_time_integration; int particles_time_integration_num;

	int particles_steps;


public:
	// arrays are hard to deal with methods in c++, so i make them public
	double particles_tau[100];

	// main functions
	SettingsCMM(std::string sim_name, int gridCoarse, int gridFine, std::string initialCondition);
	SettingsCMM();
	SettingsCMM(int argc, char *args[]);

	// function to set all values to preset ones
	void setPresets();
	// function to apply commands from command line
	void applyCommands(int argc, char *args[]);
	// helper functions
	bool getBoolFromString(std::string value);
	// work with arrays
	void string_to_int_array(std::string s_array, int *array);
	void string_to_double_array(std::string s_array, double *array);
	void transcribe_int_array(int *array_in, int *array_save, int num) { for (int i_num = 0; i_num < num; ++i_num) array_save[i_num] = array_in[i_num]; }
	void transcribe_double_array(double *array_in, double *array_save, int num) { for (int i_num = 0; i_num < num; ++i_num) array_save[i_num] = array_in[i_num]; }

	// get and set methods for all variables, this is quite much and quite ugly, i know


	// name
	std::string getWorkspace() const { return workspace; }
	void setWorkspace(std::string Workspace) { workspace = Workspace; }
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
	double getSnapshotsPerSec() const { return snapshots_per_sec; }
	void setSnapshotsPerSec(double snapshotsPerSec) { snapshots_per_sec = snapshotsPerSec; }
	bool getSetDtBySteps() const { return set_dt_by_steps; }
	void setSetDtBySteps(bool setDtBySteps) { set_dt_by_steps = setDtBySteps; }

	bool getSaveInitial() const { return save_initial; }
	void setSaveInitial(bool saveInitial) { save_initial = saveInitial; }
	bool getSaveFinal() const { return save_final; }
	void setSaveFinal(bool saveFinal) { save_final = saveFinal; }
	bool getConvInitFinal() const { return conv_init_final; }
	void setConvInitFinal(bool convInitFinal) { conv_init_final = convInitFinal; }

	// initial conditions setting
	std::string getInitialCondition() const { return initial_condition; }
	void setInitialCondition(std::string initialCondition) {
		initial_condition = initialCondition;
		// tied to num, for faster handling
		if(initialCondition == "4_nodes") initial_condition_num = 0;
		else if(initialCondition == "quadropole") initial_condition_num = 1;
		else if(initialCondition == "two_votices") initial_condition_num = 2;
		else if(initialCondition == "three_vortices") initial_condition_num = 3;
		else if(initialCondition == "single_shear_layer") initial_condition_num = 4;
		else if(initialCondition == "turbulence_gaussienne") initial_condition_num = 5;
		else if(initialCondition == "shielded_vortex") initial_condition_num = 6;
		else initial_condition_num = -1;
	}
	int getInitialConditionNum() const { return initial_condition_num; }

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
	bool getMapUpdateGrid() const { return map_update_grid; }
	void setMapUpdateGrid(bool mapUpdateGrid) { map_update_grid = mapUpdateGrid; }

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

	// skip remapping
	bool getSkipRemapping() const { return skip_remapping; }
	void setSkipRemapping(bool skipRemapping) { skip_remapping = skipRemapping; }


	// possibility to sample on specific grid
	bool getSampleOnGrid() const { return sample_on_grid; }
	void setSampleOnGrid(bool sampleOnGrid) { sample_on_grid = sampleOnGrid; }
	int getGridSample() const { return grid_sample; }
	void setGridSample(int gridSample) { grid_sample = gridSample; }
	double getSampleSnapshotsPerSec() const { return sample_snapshots_per_sec; }
	void setSampleSnapshotsPerSec(double sampleSnapshotsPerSec) { sample_snapshots_per_sec = sampleSnapshotsPerSec; }

	bool getSampleSaveInitial() const { return sample_save_initial; }
	void setSampleSaveInitial(bool sampleSaveInitial) { sample_save_initial = sampleSaveInitial; }
	bool getSampleSaveFinal() const { return sample_save_final; }
	void setSampleSaveFinal(bool sampleSaveFinal) { sample_save_final = sampleSaveFinal; }


	// zoom
	bool getZoom() const { return zoom; }
	void setZoom(bool Zoom) { zoom = Zoom; }
	int getGridZoom() const { return grid_zoom; }
	void setGridZoom(int gridZoom) { grid_zoom = gridZoom; }

	double getZoomCenterX() const { return zoom_center_x; }
	void setZoomCenterX(double zoomCenterX) { zoom_center_x = zoomCenterX; }
	double getZoomCenterY() const { return zoom_center_y; }
	void setZoomCenterY(double zoomCenterY) { zoom_center_y = zoomCenterY; }
	double getZoomWidthX() const { return zoom_width_x; }
	void setZoomWidthX(double zoomWidthX) { zoom_width_x = zoomWidthX; }
	double getZoomWidthY() const { return zoom_width_y; }
	void setZoomWidthY(double zoomWidthY) { zoom_width_y = zoomWidthY; }

	int getZoomRepetitions() const { return zoom_repetitions; }
	void setZoomRepetitions(int zoomRepetitions) { zoom_repetitions = zoomRepetitions; }
	double getZoomRepetitionsFactor() const { return zoom_repetitions_factor; }
	void setZoomRepetitionsFactor(double zoomRepetitionsFactor) { zoom_repetitions_factor = zoomRepetitionsFactor; }

	bool getZoomSavePsi() const { return zoom_save_psi; }
	void setZoomSavePsi(bool zoomSavePsi) { zoom_save_psi = zoomSavePsi; }
	bool getZoomSaveParticles() const { return zoom_save_particles; }
	void setZoomSaveParticles(bool zoomSaveParticles) { zoom_save_particles = zoomSaveParticles; }

	double getZoomSnapshotsPerSec() const { return zoom_snapshots_per_sec; }
	void setZoomSnapshotsPerSec(double zoomSnapshotsPerSec) { zoom_snapshots_per_sec = zoomSnapshotsPerSec; }

	bool getZoomSaveInitial() const { return zoom_save_initial; }
	void setZoomSaveInitial(bool zoomSaveInitial) { zoom_save_initial = zoomSaveInitial; }
	bool getZoomSaveFinal() const { return zoom_save_final; }
	void setZoomSaveFinal(bool zoomSaveFinal) { zoom_save_final = zoomSaveFinal; }


	// particles
	bool getParticles() const { return particles; }
	void setParticles(bool Particles) { particles = Particles; }
	int getParticlesNum() const { return particles_num; }
	void setParticlesNum(int particlesNum) { particles_num = particlesNum; }

	unsigned long long getParticlesSeed() const { return particles_seed; }
	void setParticlesSeed(unsigned long long particlesSeed) { particles_seed = particlesSeed; }

	double getParticlesCenterX() const { return particles_center_x; }
	void setParticlesCenterX(double particlesCenterX) { particles_center_x = particlesCenterX; }
	double getParticlesCenterY() const { return particles_center_y; }
	void setParticlesCenterY(double particlesCenterY) { particles_center_y = particlesCenterY; }
	double getParticlesWidthX() const { return particles_width_x; }
	void setParticlesWidthX(double particlesWidthX) { particles_width_x = particlesWidthX; }
	double getParticlesWidthY() const { return particles_width_y; }
	void setParticlesWidthY(double particlesWidthY) { particles_width_y = particlesWidthY; }

	int getParticlesTauNum() const { return particles_tau_num; }
	void setParticlesTauNum(int particlesTauNum) { particles_tau_num = particlesTauNum; }

	double getParticlesSnapshotsPerSec() const { return particles_snapshots_per_sec; }
	void setParticlesSnapshotsPerSec(double particlesSnapshotsPerSec) { particles_snapshots_per_sec = particlesSnapshotsPerSec; }

	bool getParticlesSaveInitial() const { return particles_save_initial; }
	void setParticlesSaveInitial(bool particlesSaveInitial) { particles_save_initial = particlesSaveInitial; }
	bool getParticlesSaveFinal() const { return particles_save_final; }
	void setParticlesSaveFinal(bool particlesSaveFinal) { particles_save_final = particlesSaveFinal; }

	// fine particles
	bool getSaveFineParticles() const { return save_fine_particles; }
	void setSaveFineParticles(bool saveFineParticles) { save_fine_particles = saveFineParticles; }
	int getParticlesFineNum() const { return particles_fine_num; }
	void setParticlesFineNum(int particlesFineNum) { particles_fine_num = particlesFineNum; }

	// particle time integration
	std::string getParticlesTimeIntegration() const { return particles_time_integration; }
	void setParticlesTimeIntegration(std::string pTimeIntegration) {
		particles_time_integration = pTimeIntegration;
		if (pTimeIntegration == "EulerExp") { particles_time_integration_num = 10; if (getLagrangeOrder() < 1) lagrange_order = 1; }
		else if (pTimeIntegration == "Heun") { particles_time_integration_num = 20; if (getLagrangeOrder() < 2) lagrange_order = 2; }
		else if (pTimeIntegration == "RK3") { particles_time_integration_num = 30; if (getLagrangeOrder() < 3) lagrange_order = 3; }
		else if (pTimeIntegration == "RK4") { particles_time_integration_num = 40; if (getLagrangeOrder() < 4) lagrange_order = 4; }
		else if (pTimeIntegration == "RK3Mod") { particles_time_integration_num = 31; if (getLagrangeOrder() < 3) lagrange_order = 3; }
		else if (pTimeIntegration == "RK4Mod") { particles_time_integration_num = 41; if (getLagrangeOrder() < 4) lagrange_order = 4; }
		else particles_time_integration_num = -1;
	}
	int getParticlesTimeIntegrationNum() const { return particles_time_integration_num; }

	int getParticlesSteps() const { return particles_steps; }
	void setParticlesSteps(int particlesSteps) { particles_steps = particlesSteps; }
};

#endif
