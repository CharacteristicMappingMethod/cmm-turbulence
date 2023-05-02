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

#include "cmm-param.h"
#include "../ui/settings.h"
#include "../ui/cmm-io.h"
#include "string.h"
#include <iostream>
#include "fstream"

/*
 * parameter file version, which should be increased with every major new setting
 * maybe this is not important, however, as older parameter files could leave out settings, they might not work anymore afterwards
 */
const int param_version = 3;
const std::string param_header_line = "CMM Parameter file";
const std::string param_command = "param_version";
const std::string param_delimiter = "\t=\t";

// because I have to use this many times, this just makes it easier and more readable
std::string write_line(std::string param_name, std::string param_value) {
	std::ostringstream os;
	os << param_name << param_delimiter << param_value << "\n";
	return os.str ();
}


// function to safe parameter file, saved is within simulation data scope with fixed name, pass-by-reference
void save_param_file(SettingsCMM& SettingsMain, std::string param_name) {
	std::ofstream file;

//	std::string file_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/params.txt";
	file.open(param_name.c_str(), std::ios::out);

	if(!file)
	{
		std::cout<<"Unable to create parameter file.\n";
	}
	else
	{
		// first three lines as setting lines for some details
		file << param_header_line << "\n";  // header
		file << write_line(param_command, to_str(param_version, 16));  // give it a version number
		file << "\n";  // empty dummy line, for now as a separator, but maybe it comes in handy later

		// now come all the parameter!
		file << write_line("workspace", to_str(SettingsMain.getWorkspace(), 16));
		file << write_line("simulation_type", to_str(SettingsMain.getSimulationType(), 16));
		file << write_line("sim_name", to_str(SettingsMain.getSimName(), 16));

		// grid details
		file << write_line("grid_coarse", to_str(SettingsMain.getGridCoarse(), 16));
		file << write_line("grid_fine", to_str(SettingsMain.getGridFine(), 16));
		file << write_line("grid_psi", to_str(SettingsMain.getGridPsi(), 16));
		file << write_line("grid_vort", to_str(SettingsMain.getGridVort(), 16));

		// time details
		file << write_line("final_time", to_str(SettingsMain.getFinalTime(), 16));
		file << write_line("set_dt_by_steps", to_str(SettingsMain.getSetDtBySteps(), 16));
		if (SettingsMain.getSetDtBySteps()) {
			file << write_line("steps_per_sec", to_str(SettingsMain.getStepsPerSec(), 16));
		}
		else {
			file << write_line("factor_dt_by_grid", to_str(SettingsMain.getFactorDtByGrid(), 16));
		}
		file << write_line("save_computational_num", to_str(SettingsMain.getSaveComputationalNum(), 16));
		for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
			file << write_line("save_computational", to_str(SettingsMain.getSaveComputational()[i_save].getVariables(), 16));
		}

		file << write_line("mem_RAM_CPU_remaps", to_str(SettingsMain.getMemRamCpuRemaps(), 16));
		file << write_line("save_map_stack", to_str(SettingsMain.getSaveMapStack(), 16));
		file << write_line("restart_time", to_str(SettingsMain.getRestartTime(), 16));
		if (SettingsMain.getRestartTime() != 0) {
			file << write_line("restart_location", to_str(SettingsMain.getRestartLocation(), 16));
		}

		file << write_line("verbose", to_str(SettingsMain.getVerbose(), 16));

		file << write_line("initial_condition", to_str(SettingsMain.getInitialCondition(), 16));
		file << write_line("initial_params", to_str(SettingsMain.getInitialParams(), 16));
		file << write_line("initial_discrete", to_str(SettingsMain.getInitialDiscrete(), 16));
		if (SettingsMain.getInitialDiscrete())  {
			file << write_line("initial_discrete_grid", to_str(SettingsMain.getInitialDiscreteGrid(), 16));
			file << write_line("initial_discrete_location", to_str(SettingsMain.getInitialDiscreteLocation(), 16));
		}

		file << write_line("incomp_threshold", to_str(SettingsMain.getIncompThreshold(), 16));
		file << write_line("map_epsilon", to_str(SettingsMain.getMapEpsilon(), 16));
		file << write_line("time_integration", to_str(SettingsMain.getTimeIntegration(), 16));
		file << write_line("lagrange_override", to_str(SettingsMain.getLagrangeOverride(), 16));
		file << write_line("lagrange_init_higher_order", to_str(SettingsMain.getLagrangeInitHigherOrder(), 16));
		file << write_line("map_update_order", to_str(SettingsMain.getMapUpdateOrder(), 16));
		file << write_line("molly_stencil", to_str(SettingsMain.getMollyStencil(), 16));
		file << write_line("freq_cut_psi", to_str(SettingsMain.getFreqCutPsi(), 16));
		file << write_line("skip_remapping", to_str(SettingsMain.getSkipRemapping(), 16));

		file << write_line("save_sample_num", to_str(SettingsMain.getSaveSampleNum(), 16));
		for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
			file << write_line("save_sample", to_str(SettingsMain.getSaveSample()[i_save].getVariables(), 16));
		}

		file << write_line("scalar_name", to_str(SettingsMain.getScalarName(), 16));
		file << write_line("scalar_discrete", to_str(SettingsMain.getScalarDiscrete(), 16));
		if (SettingsMain.getScalarDiscrete()) {
			file << write_line("scalar_discrete_grid", to_str(SettingsMain.getScalarDiscreteGrid(), 16));
			file << write_line("scalar_discrete_location", to_str(SettingsMain.getScalarDiscreteLocation(), 16));
		}

		file << write_line("save_zoom_num", to_str(SettingsMain.getSaveZoomNum(), 16));
		for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
			file << write_line("save_zoom", to_str(SettingsMain.getSaveZoom()[i_save].getVariables(), 16));
		}

		file << write_line("forward_map", to_str(SettingsMain.getForwardMap(), 16));
		if (SettingsMain.getForwardMap()) {
			file << write_line("particles_forwarded_num", to_str(SettingsMain.getParticlesForwardedNum(), 16));
			for (int i_particles = 0; i_particles < SettingsMain.getParticlesForwardedNum(); ++i_particles) {
				file << write_line("particles_forwarded", to_str(SettingsMain.getParticlesForwarded()[i_particles].getVariables(), 16));
			}
		}

		file << write_line("particles_advected_num", to_str(SettingsMain.getParticlesAdvectedNum(), 16));
		for (int i_particles = 0; i_particles < SettingsMain.getParticlesAdvectedNum(); ++i_particles) {
			file << write_line("particles_advected", to_str(SettingsMain.getParticlesAdvected()[i_particles].getVariables(), 16));
		}

		if (SettingsMain.getParticlesAdvectedNum() > 0) {
			// hackery for particle convergence
			file << write_line("particles_steps", to_str(SettingsMain.getParticlesSteps(), 16));
		}

		file.close();
	}
}

// function to load parameter file, pass-by-reference
void load_param_file(SettingsCMM& SettingsMain, std::string param_name) {
	std::ifstream file;

	file.open(param_name.c_str(), std::ios::in);

	if(!file)
	{
		std::cout<<"Unable to read parameter file.\n";
	}
	else
	{
		std::string file_line;
		int param_version;

		// check first line, this has to be equal to be accepted as param file
		getline(file, file_line);
		if (file_line == param_header_line) {
			// check for version with next line
			getline(file, file_line);
			int pos_equal = file_line.find(param_delimiter);
			if (pos_equal != std::string::npos) {
				// construct two substrings
				std::string command = file_line.substr(0, pos_equal);
				std::string value = file_line.substr(pos_equal+param_delimiter.length(), file_line.length());

				if (command == param_command) {
					param_version = std::stoi(value);

					// skip third line
					getline(file, file_line);

					// now read in all the values
					while(getline(file, file_line)){ //read data from file object and put it into string.
						int set_save = SettingsMain.setVariable(file_line, param_delimiter);
						if (set_save == 1) {
							for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
								getline(file, file_line);
								SettingsMain.setSaveComputational(file_line, param_delimiter, i_save);
							}
						}
						else if (set_save == 2) {
							for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
								getline(file, file_line);
								SettingsMain.setSaveSample(file_line, param_delimiter, i_save);
							}
						}
						else if (set_save == 3) {
							for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
								getline(file, file_line);
								SettingsMain.setSaveZoom(file_line, param_delimiter, i_save);
							}
						}
						else if (set_save == 4) {
							for (int i_save = 0; i_save < SettingsMain.getParticlesAdvectedNum(); ++i_save) {
								getline(file, file_line);
								SettingsMain.setParticlesAdvected(file_line, param_delimiter, i_save);
							}
						}
						else if (set_save == 5) {
							for (int i_save = 0; i_save < SettingsMain.getParticlesForwardedNum(); ++i_save) {
								getline(file, file_line);
								SettingsMain.setParticlesForwarded(file_line, param_delimiter, i_save);
							}
						}
					}

					// here, future settings for newer param-file versions could be implemented for example to disable stuff by default
					if (param_version < 2) {
						std::cout<<"Param-file version < 2: Save settings cannot be set." << std::endl;
					}
					if (param_version < 3) {
						std::cout<<"Param-file version < 3: Initial condition parameters cannot be set."
							"Check cmm-init.cu if the initial conditions contains parameters that need to be set" << std::endl;
					}
				}
			}
		}

		file.close(); //close the file object.
	}
}
