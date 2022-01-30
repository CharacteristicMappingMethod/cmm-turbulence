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
#define PARAM_VERSION 2
#define PARAM_HEADER_LINE "CMM Parameter file"
#define PARAM_COMMAND "param_version"

// because I have to use this many times, this just makes it easier and more readable
std::string write_line(std::string param_name, std::string param_value) {
	ostringstream os;
	os << param_name << "\t=\t" << param_value << "\n";
	return os.str ();
}


// function to safe parameter file, saved is within simulation data scope with fixed name, pass-by-reference
void save_param_file(SettingsCMM& SettingsMain, std::string param_name) {
	std::ofstream file;

//	std::string file_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/params.txt";
	file.open(param_name.c_str(), ios::out);

	if(!file)
	{
		std::cout<<"Unable to create parameter file.\n";
	}
	else
	{
		// first three lines as setting lines for some details
		file << PARAM_HEADER_LINE << "\n";  // header
		file << write_line(PARAM_COMMAND, to_str(PARAM_VERSION, 16));  // give it a version number
		file << "\n";  // empty dummy line, for now as a separator, but maybe it comes in handy later

		// now come all the parameter!
		file << write_line("workspace", to_str(SettingsMain.getWorkspace(), 16));
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

		file << write_line("verbose", to_str(SettingsMain.getVerbose(), 16));

		file << write_line("initial_condition", to_str(SettingsMain.getInitialCondition(), 16));
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
		file << write_line("map_update_grid", to_str(SettingsMain.getMapUpdateGrid(), 16));
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
	ifstream file;

	file.open(param_name.c_str(), ios::in);

	if(!file)
	{
		std::cout<<"Unable to read parameter file.\n";
	}
	else
	{
		std::string file_line;
		int param_version;
		std::string delimiter = "\t=\t";

		// check first line, this has to be equal to be accepted as param file
		getline(file, file_line);
		if (file_line == PARAM_HEADER_LINE) {
			// check for version with next line
			getline(file, file_line);
			int pos_equal = file_line.find(delimiter);
			if (pos_equal != std::string::npos) {
				// construct two substrings
				std::string command = file_line.substr(0, pos_equal);
				std::string value = file_line.substr(pos_equal+delimiter.length(), file_line.length());

				if (command == PARAM_COMMAND) {
					param_version = std::stoi(value);

					// skip third line
					getline(file, file_line);

					// now read in all the values
					while(getline(file, file_line)){ //read data from file object and put it into string.
						int set_save = SettingsMain.setVariable(file_line, delimiter);
						if (set_save == 1) {
							for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
								SettingsMain.setSaveComputational(file_line, delimiter, i_save);
							}
						}
						else if (set_save == 2) {
							for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
								SettingsMain.setSaveSample(file_line, delimiter, i_save);
							}
						}
						else if (set_save == 3) {
							for (int i_save = 0; i_save < SettingsMain.getSaveZoomNum(); ++i_save) {
								SettingsMain.setSaveZoom(file_line, delimiter, i_save);
							}
						}
						else if (set_save == 4) {
							for (int i_save = 0; i_save < SettingsMain.getParticlesAdvectedNum(); ++i_save) {
								SettingsMain.setParticlesAdvected(file_line, delimiter, i_save);
							}
						}
						else if (set_save == 5) {
							for (int i_save = 0; i_save < SettingsMain.getParticlesForwardedNum(); ++i_save) {
								SettingsMain.setParticlesForwarded(file_line, delimiter, i_save);
							}
						}
					}

					// here, future settings for newer param-file versions could be implemented for example to disable stuff by default
					if (param_version < 2) {
						std::cout<<"Tried to read in old file. Save settings cannot be set.\n";
					}
				}
			}
		}

		file.close(); //close the file object.
	}
}
