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

#include "cmm-io.h"

#include "../simulation/cmm-simulation-kernel.h"
#include "../numerical/cmm-mesure.h"  // used for hashing

/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

void create_directory_structure(SettingsCMM SettingsMain, double dt, int iterMax)
{
	std::string folder_data = SettingsMain.getWorkspace() + "data";
	struct stat st = {0};
	if (stat(folder_data.c_str(), &st) == -1) mkdir(folder_data.c_str(), 0777);

	//creating main folder
	std::string folder_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName();
	mkdir(folder_name.c_str(), 0777);

	// delete monitoring data, only if we do not restart, cause then its appended
	if (SettingsMain.getRestartTime() == 0) {
		int del = recursive_delete(folder_name + "/Monitoring_data", 0);
	}
	// create general subfolder for other data
	std::string folder_name_tdata = folder_name + "/Monitoring_data";
	mkdir(folder_name_tdata.c_str(), 0777);
	// create general subfolder for mesure data
	folder_name_tdata = folder_name + "/Monitoring_data/Mesure";
	mkdir(folder_name_tdata.c_str(), 0777);
	// some stats are computed for debugging purposes
	if (SettingsMain.getVerbose() >= 4) {
		folder_name_tdata = folder_name + "/Monitoring_data/Debug_globals";
		mkdir(folder_name_tdata.c_str(), 0777);
	}

	// create general subfolder for timesteps
	folder_name_tdata = folder_name + "/Time_data";
	mkdir(folder_name_tdata.c_str(), 0777);

	// create general subfolder for zoom
	if (SettingsMain.getSaveZoomNum() > 0) {
		folder_name_tdata = folder_name + "/Zoom_data";
		mkdir(folder_name_tdata.c_str(), 0777);
	}

	// create general subfolder for particles
	if (SettingsMain.getParticlesAdvectedNum() > 0 || SettingsMain.getParticlesForwardedNum() > 0) {
		folder_name_tdata = folder_name + "/Particle_data";
		mkdir(folder_name_tdata.c_str(), 0777);
	}
}


/*******************************************************************
*					    Writting in binary						   *
*******************************************************************/


void writeAllRealToBinaryFile(size_t Len, double *var, SettingsCMM SettingsMain, std::string data_name)
{
	std::string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	std::ofstream file(fileName.c_str(), std::ios::out | std::ios::binary);

	if(!file)
	{
		std::cout<<"Error saving file. Unable to open : "<<fileName<< std::endl;
		return;
	}
	else {
		file.write( (char*) var, Len*sizeof(double) );
	}

	file.close();
}
void writeAppendToBinaryFile(size_t Len, double *var, SettingsCMM SettingsMain, std::string data_name)
{
	std::string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	std::ofstream file(fileName.c_str(), std::ios::out | std::ios::app | std::ios::binary);

	if(!file)
	{
		std::cout<<"Error saving file. Unable to open : "<<fileName<< std::endl;
		return;
	}
	else {
		file.write( (char*) var, Len*sizeof(double) );
	}

	file.close();
}


bool readAllRealFromBinaryFile(size_t Len, double *var, std::string data_name)
{
	std::string fileName = data_name;
	std::ifstream file(fileName.c_str(), std::ios::in | std::ios::binary);
	bool open_file;

	if(!file)
	{
		std::cout<<"Error reading file. Unable to open : "<<fileName<< std::endl;
		open_file = false;
	}
	else {
		file.read( (char*) var, Len*sizeof(double) );
		open_file = true;
	}

	file.close();
	return open_file;
}



void writeTransferToBinaryFile(size_t Len, double *d_var, SettingsCMM SettingsMain, std::string data_name, bool do_append)  {
	size_t chunkSize = 1024;
	std::string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	std::ofstream file;
	if (!do_append) file.open(fileName.c_str(), std::ios::out | std::ios::binary);
	else file.open(fileName.c_str(), std::ios::out | std::ios::app | std::ios::binary);

	if(!file)
	{
		std::cout<<"Error saving file. Unable to open : "<<fileName<< std::endl;
		return;
	}
	else {
		// Create a stream
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		// Read and transfer the data in chunks
		size_t offset = 0;
		while (offset < Len) {
			// check to match Len perfectly, I had to remove abs function, but it should work without
			if (Len - offset < chunkSize) chunkSize = Len - offset;

			// Allocate host memory for the chunk
			double* hostData = new double[chunkSize];

			// Transfer the chunk from host to device asynchronously
			cudaMemcpyAsync(hostData, d_var + offset, chunkSize*sizeof(double), cudaMemcpyDeviceToHost, stream);

			// Read the chunk from the file
			file.write((char*)hostData, chunkSize*sizeof(double));

			// Free the host memory
			delete[] hostData;

			offset += chunkSize;
		}

		// Synchronize and destroy the stream
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}

	file.close();
}
bool readTransferFromBinaryFile(size_t Len, double *d_var, std::string data_name) {
	size_t chunkSize = 1024;
	std::string fileName = data_name;
	std::ifstream file(fileName.c_str(), std::ios::in | std::ios::binary);
	bool open_file;

	if(!file)
	{
		std::cout<<"Error reading file. Unable to open : "<<fileName<< std::endl;
		open_file = false;
	}
	else {
		// Create a stream
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		// Read and transfer the data in chunks
		long long int offset = 0;
		while (!file.eof() && offset < Len) {
			// check to match Len perfectly, I had to remove abs function, but it should work without
			if (Len - offset < chunkSize) chunkSize = Len - offset;

			// Allocate host memory for the chunk
			double* hostData = new double[chunkSize];

			// Read the chunk from the file
			file.read((char*)hostData, chunkSize*sizeof(double));

			// Transfer the chunk from host to device asynchronously
			cudaMemcpyAsync(d_var + offset, hostData, chunkSize*sizeof(double), cudaMemcpyHostToDevice, stream);

			// Free the host memory
			delete[] hostData;

			offset += chunkSize;
		}

		// Synchronize and destroy the stream
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);

		open_file = true;
	}

	file.close();
	return open_file;
}


/*******************************************************************
*					    File handling							   *
*******************************************************************/
// C++17 has a library dealing with that but project needs C++14 as well
int recursive_delete(std::string path, int debug) {
    DIR *dir; struct dirent *entry; std::string child_path;

    /* Open the directory */
    dir = opendir(path.c_str());
    if (!dir) {
        if (debug > 0) perror("Error opening directory");
        return 0;
    }

    /* Recursively delete all of the directory's contents */
    while ((entry = readdir(dir))) {
        child_path = path + "/" + entry->d_name;
        if (entry->d_type == DT_DIR) {
            // check for . or ..
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            // next recursion
            if (recursive_delete(std::string(child_path), debug) == 0) {
                closedir(dir); return 0;
            }
        }
        // remove file
        else {
            if (debug > 1) std::cout << "Trying to delete file " << child_path << std::endl;
            else {
                if (remove(child_path.c_str()) != 0) {
                    if (debug > 0) perror("Error deleting file");
                    closedir(dir); return 0;
                }
            }
        }
    }
    closedir(dir);

    // remove folder itself
    if (debug > 1) std::cout << "Trying to delete folder " << path << std::endl;
    else {
        if (remove(path.c_str()) != 0) {
            if (debug > 0) perror("Error deleting directory");
            return 0;
        }
    }

    if (debug) return -1;
    else return 1;
}


/*******************************************************************
* Structures to create or save on timestep in either hdf5 or binary
*
* hdf5: create subgroup for the timestep and save values there
* add attributes to group
*
* binary: create subfolder for the timestep and save values there
* attributes are not directly given, maybe over a readme file in folder
*******************************************************************/

// hdf5 version
#ifdef HDF5_INCLUDE
std::string writeTimeStep(SettingsCMM SettingsMain, std::string i_num, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_W_coarse, double *Dev_Psi_real,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f) {
	}

// binary version
#else
	std::string writeTimeStep(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
			std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap, double *Dev_Temp) {

		// check if we want to save at this time, combine all variables if so
		bool save_now = false; std::string save_var;
		SaveComputational* save_comp = SettingsMain.getSaveComputational();
		for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
			// instants - distance to target is smaller than threshhold
			if (save_comp[i_save].is_instant && t_now - save_comp[i_save].time_start + dt*1e-5 < dt_now && t_now - save_comp[i_save].time_start + dt*1e-5 >= 0) {
				save_now = true; save_var = save_var + save_comp[i_save].var;
			}
			// intervals - modulo to steps with safety-increased targets is smaller than step
			if (!save_comp[i_save].is_instant
				&& ((fmod(t_now - save_comp[i_save].time_start + dt*1e-5, save_comp[i_save].time_step) < dt_now
				&& t_now + dt*1e-5 >= save_comp[i_save].time_start
				&& t_now - dt*1e-5 <= save_comp[i_save].time_end)
				|| t_now == save_comp[i_save].time_end)) {
				save_now = true; save_var = save_var + save_comp[i_save].var;
			}
		}
		std::string message = ""; bool saved_any_var = false;
		if (save_now) {
			std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
			std::string sub_folder_name = "";
			bool save_all = save_var.find("All") != std::string::npos;  // master save switch
			if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos or
					save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos or
					save_var.find("Map") != std::string::npos or save_var.find("Chi") != std::string::npos or
					save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos or
					save_var.find("Dist") != std::string::npos or save_var.find("F") != std::string::npos or
					save_all) {
				// create new subfolder for current timestep
				sub_folder_name = "/Time_data/Time_" + t_s_now;
				std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
				struct stat st = {0};
				if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

				// for debug: save time position
				if (SettingsMain.getVerbose() >= 4) {
					writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Debug_globals/Time_s");  // time vector for data
				}
				saved_any_var = true;  // so that we dont save timing twice in particles

				// construct message here as we are sure it is being saved something
				message = "Processed computational data";
			}

			// Vorticity on coarse grid : W_coarse
			if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos or save_all) {
				writeTimeVariable(SettingsMain, "Vorticity_W_coarse", t_now, *cmmVarMap["Vort"]);
			}

			// Distribution function on computational grid, is just Vort currently
			if (save_var.find("Dist") != std::string::npos or save_var.find("F") != std::string::npos or save_all) {
				writeTimeVariable(SettingsMain, "Distribution_coarse", t_now, *cmmVarMap["Vort"]);
			}

			// Stream function on psi grid : Psi_psi
			if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos or save_all) {
				writeTimeVariable(SettingsMain, "Stream_function_Psi_psi", t_now, *cmmVarMap["Psi"]);
			}
			if (save_var.find("Stream_H") != std::string::npos or save_var.find("Psi_H") != std::string::npos or save_all) {
				writeArr(SettingsMain, sub_folder_name, "Stream_function_Psi_H_psi", cmmVarMap["Psi"]->Dev_var, 4*cmmVarMap["Psi"]->Grid->N);
//				writeTransferToBinaryFile(4*cmmVarMap["Psi"]->Grid->N, cmmVarMap["Psi"]->Dev_var, SettingsMain, sub_folder_name + "/Stream_function_Psi_H_psi", false);
			}

			// Velocity on psi grid : U_psi, x-direction and y-direction
			if (save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos or save_all) {
				writeTimeVariable(SettingsMain, "Velocity_UX_psi", t_now, *cmmVarMap["Psi"], 1*cmmVarMap["Psi"]->Grid->N);
				writeTimeVariable(SettingsMain, "Velocity_UY_psi", t_now, *cmmVarMap["Psi"], 2*cmmVarMap["Psi"]->Grid->N);
			}

			// Backwards map on coarse grid in Hermite or single version : Chi_coarse, x- and y-direction
			if (save_var.find("Map_b") != std::string::npos or save_var.find("Chi_b") != std::string::npos or save_all) {
				writeTimeVariable(SettingsMain, "Map_ChiX_coarse", t_now, *cmmVarMap["ChiX"]);
				writeTimeVariable(SettingsMain, "Map_ChiY_coarse", t_now, *cmmVarMap["ChiY"]);
			}
			if (save_var.find("Map_H") != std::string::npos or save_var.find("Chi_H") != std::string::npos or save_all) {
				writeArr(SettingsMain, sub_folder_name, "Map_ChiX_H_coarse", cmmVarMap["ChiX"]->Dev_var, 4*cmmVarMap["ChiX"]->Grid->N);
				writeArr(SettingsMain, sub_folder_name, "Map_ChiY_H_coarse", cmmVarMap["ChiY"]->Dev_var, 4*cmmVarMap["ChiY"]->Grid->N);
//				writeTransferToBinaryFile(4*cmmVarMap["ChiX"]->Grid->N, cmmVarMap["ChiX"]->Dev_var, SettingsMain, sub_folder_name + "/Map_ChiX_H_coarse", false);
//				writeTransferToBinaryFile(4*cmmVarMap["ChiY"]->Grid->N, cmmVarMap["ChiY"]->Dev_var, SettingsMain, sub_folder_name + "/Map_ChiY_H_coarse", false);
			}

			// Forwards map on coarse grid in Hermite or single version : Chi_f_coarse, x- and y-direction
			if (SettingsMain.getForwardMap() and (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos or save_all)) {
				writeTimeVariable(SettingsMain, "Map_ChiX_f_coarse", t_now, *cmmVarMap["ChiX_f"]);
				writeTimeVariable(SettingsMain, "Map_ChiY_f_coarse", t_now, *cmmVarMap["ChiY_f"]);
			}
			if (SettingsMain.getForwardMap() and (save_var.find("Map_f_H") != std::string::npos or save_var.find("Chi_f_H") != std::string::npos or save_all)) {
				writeArr(SettingsMain, sub_folder_name, "Map_ChiX_f_H_coarse", cmmVarMap["ChiX_f"]->Dev_var, 4*cmmVarMap["ChiX_f"]->Grid->N);
				writeArr(SettingsMain, sub_folder_name, "Map_ChiY_f_H_coarse", cmmVarMap["ChiY_f"]->Dev_var, 4*cmmVarMap["ChiY_f"]->Grid->N);
//				writeTransferToBinaryFile(4*cmmVarMap["ChiX_f"]->Grid->N, cmmVarMap["ChiX_f"]->Dev_var, SettingsMain, sub_folder_name + "/Map_ChiX_f_H_coarse", false);
//				writeTransferToBinaryFile(4*cmmVarMap["ChiY_f"]->Grid->N, cmmVarMap["ChiY_f"]->Dev_var, SettingsMain, sub_folder_name + "/Map_ChiY_f_H_coarse", false);
			}

			/*
			 *  particles, it was moved here because it is also controlled with saveComputational
			 */
			// save for every PartA_ that is found
			ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();

			sub_folder_name = "/Particle_data/Time_" + t_s_now;
			std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;

			// create new subfolder for current timestep, doesn't matter if we try to create it several times
			if (save_var.find("PartA_", 0) != std::string::npos or save_all) {
				struct stat st = {0};
				if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

				if (!saved_any_var and SettingsMain.getVerbose() >= 4) {
					writeAppendToBinaryFile(1, &t_now, SettingsMain, "/Monitoring_data/Debug_globals/Time_s");  // time vector for data
				}

				// construct message here as we are sure it is being saved something
				message = "Processed particle data";
			}

			for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
				// particle position first
				if (save_var.find("PartA_" + to_str_0(i_p+1, 2)) != std::string::npos or save_all) {
					std::string part_map_name = "PartA_Pos_P" + to_str_0(i_p+1, 2);
					writePart(SettingsMain, sub_folder_name, "/Particles_advected_pos_P" + to_str_0(i_p+1, 2), *cmmPartMap[part_map_name]);
				}
				// particle velocity now
				if (save_var.find("PartA_Vel_" + to_str_0(i_p+1, 2)) != std::string::npos or save_all) {
					// copy data to host and save - but only if tau != 0
					if (particles_advected[i_p].tau != 0) {
						std::string part_map_name = "PartA_Vel_U" + to_str_0(i_p+1, 2);
						writePart(SettingsMain, sub_folder_name, "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), *cmmPartMap[part_map_name]);
					}
					// sample velocity values from current stream function at particle positions
					else {
						std::string part_map_name = "PartA_Pos_P" + to_str_0(i_p+1, 2);
						CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
						k_h_sample_points_dxdy<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*cmmVarMap["Psi"]->Grid, *cmmVarMap["Psi"]->Grid, cmmVarMap["Psi"]->Dev_var, Part_Pos_now.Dev_var, Dev_Temp, Part_Pos_now.num);
						writeArr(SettingsMain, sub_folder_name, "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), Dev_Temp, 2*Part_Pos_now.num);
					}
				}
			}
		}
		return message;
	}
#endif


// script to save only one of the variables, needed because we need temporal arrays to save
// Data is located at Var.Dev_var + offset_start
void writeTimeVariable(SettingsCMM SettingsMain, std::string data_name, double t_now, CmmVar2D Var, size_t offset_start) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
	std::string sub_folder_name = "/Time_data/Time_" + t_s_now;
	std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	writeVar(SettingsMain, sub_folder_name, data_name, Var, offset_start);
}


// this function handles additionally the folder creation for zoom files
void writeZoomVariable(SettingsCMM SettingsMain, std::string data_name, int zoom_num, int zoom_rep, double t_now, CmmVar2D Var, size_t offset_start) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
	std::string sub_folder_name = "/Zoom_data/Time_" + t_s_now;
	std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);
	// extra folder for specific zoom an repetition
	std::string zr_s_now = "Zoom_" + to_str(zoom_num + 1) + "_rep_" + to_str(zoom_rep);
	sub_folder_name += "/" + zr_s_now; folder_name_now += "/" + zr_s_now;
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	writeVar(SettingsMain, sub_folder_name, data_name, Var, offset_start, zr_s_now);
}


void writeVar(SettingsCMM SettingsMain, std::string sub_folder_name, std::string data_name, CmmVar2D Var, size_t offset_start, std::string debug_name_add) {
	// copy and save, but not for increased debug
	if (SettingsMain.getVerbose() < 4) {
		writeTransferToBinaryFile(Var.Grid->N, Var.Dev_var+offset_start, SettingsMain, sub_folder_name + "/" + data_name, false);
	}
	else {
		std::string debug_add = "";
		if (debug_name_add != "") debug_add = "_" + debug_name_add;
		writeVarDebugGlobals(SettingsMain, data_name + debug_add, Var, offset_start);
	}
}
void writePart(SettingsCMM SettingsMain, std::string sub_folder_name, std::string data_name, CmmPart Part, size_t offset_start, std::string debug_name_add) {
	// copy and save, but not for increased debug
	if (SettingsMain.getVerbose() < 4) {
		writeTransferToBinaryFile(2*Part.num, Part.Dev_var+offset_start, SettingsMain, sub_folder_name + "/" + data_name, false);
	}
	else {
		std::string debug_add = "";
		if (debug_name_add != "") debug_add = "_" + debug_name_add;
		writeDebugGlobals(SettingsMain, data_name + debug_add, Part.Dev_var, 2*Part.num, offset_start);
	}
}
void writeArr(SettingsCMM SettingsMain, std::string sub_folder_name, std::string data_name, double* Var, size_t N, size_t offset_start, std::string debug_name_add) {
	// copy and save, but not for increased debug
	if (SettingsMain.getVerbose() < 4) {
		writeTransferToBinaryFile(N, Var+offset_start, SettingsMain, sub_folder_name + "/" + data_name, false);
	}
	else {
		std::string debug_add = "";
		if (debug_name_add != "") debug_add = "_" + debug_name_add;
		writeDebugGlobals(SettingsMain, data_name + debug_add, Var, N, offset_start);
	}
}


// save min, max, avgf and L2 of variable, CmmVar2D so grid is given
void writeVarDebugGlobals(SettingsCMM SettingsMain, std::string data_name, CmmVar2D Var, size_t offset_start) {
	double mesure[4];
	// MinMaxAvgL2
	Compute_min(mesure[0], Var, offset_start); Compute_max(mesure[1], Var, offset_start);
	Compute_avg(mesure[2], Var, offset_start); Compute_L2(mesure[3], Var, offset_start);
	writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Min");
	writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Max");
	writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Avg");
	writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_L2");
	// Hash in extra array
	double hash[2]; Hash_array((char*)hash, Var.Dev_var, Var.Grid->N);
	writeAppendToBinaryFile(2, hash, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Hash");
}
// save min, max, avgf and L2 of particles, averaged is over N
void writeDebugGlobals(SettingsCMM SettingsMain, std::string data_name, double* Var, size_t N, size_t offset_start) {
	double mesure[4];
	// MinMaxAvgL2
	Compute_min(mesure[0], Var, N, offset_start); Compute_max(mesure[1], Var, N, offset_start);
	Compute_avg(mesure[2], Var, N, offset_start); Compute_L2(mesure[3], Var, N, offset_start);
	writeAppendToBinaryFile(1, mesure, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Min");
	writeAppendToBinaryFile(1, mesure+1, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Max");
	writeAppendToBinaryFile(1, mesure+2, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Avg");
	writeAppendToBinaryFile(1, mesure+3, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_L2");
	// Hash in extra array
	double hash[2]; Hash_array((char*)hash, Var, N);
	writeAppendToBinaryFile(2, hash, SettingsMain, "/Monitoring_data/Debug_globals/" + data_name + "_Hash");
}

/*
 * Write particle positions
 */
// will be with hdf5 version too at some point
std::string writeParticles(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		std::map<std::string, CmmPart*> cmmPartMap, CmmVar2D Psi, double *Dev_Temp) {

//		double **Dev_particles_pos, double **Dev_particles_vel,
//		TCudaGrid2D Grid_psi, double *Dev_psi, double *Dev_Temp, int* fluid_particles_blocks, int fluid_particles_threads) {
	// check if we want to save at this time, combine all variables if so
	std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
	bool save_now = false; std::string save_var;
	SaveComputational* save_comp = SettingsMain.getSaveComputational();
	for (int i_save = 0; i_save < SettingsMain.getSaveComputationalNum(); ++i_save) {
		// instants - distance to target is smaller than threshhold
		if (save_comp[i_save].is_instant && t_now - save_comp[i_save].time_start + dt*1e-5 < dt_now && t_now - save_comp[i_save].time_start + dt*1e-5 >= 0) {
			save_now = true; save_var = save_var + save_comp[i_save].var;
		}
		// intervals - modulo to steps with safety-increased targets is smaller than step
		if (!save_comp[i_save].is_instant
			&& fmod(t_now - save_comp[i_save].time_start + dt*1e-5, save_comp[i_save].time_step) < dt_now
			&& t_now + dt*1e-5 >= save_comp[i_save].time_start
			&& t_now - dt*1e-5 <= save_comp[i_save].time_end) {
			save_now = true; save_var = save_var + save_comp[i_save].var;
		}
	}
	std::string message = "";
	if (save_now) {
		// save for every PartA_ that is found
		ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
		int pos_p = save_var.find("PartA_", 0);

		std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
		std::string sub_folder_name = "/Particle_data/Time_" + t_s_now;
		std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;

		// create new subfolder for current timestep, doesn't matter if we try to create it several times
		bool save_all = save_var.find("All") != std::string::npos;  // master save switch
		if (save_var.find("PartA_", 0) != std::string::npos or save_all) {
			struct stat st = {0};
			if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

			// construct message here as we are sure it is being saved something
			message = "Processed particle data";
		}

		for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
			// particle position first
			if (save_var.find("PartA_" + to_str_0(i_p+1, 2)) != std::string::npos or save_all) {
				std::string part_map_name = "PartA_Pos_P" + to_str_0(i_p+1, 2);
				writePart(SettingsMain, sub_folder_name, "/Particles_advected_pos_P" + to_str_0(i_p+1, 2), *cmmPartMap[part_map_name]);
			}
			// particle velocity now
			if (save_var.find("PartA_Vel_" + to_str_0(i_p+1, 2)) != std::string::npos or save_all) {
				// copy data to host and save - but only if tau != 0
				if (particles_advected[i_p].tau != 0) {
					std::string part_map_name = "PartA_Vel_U" + to_str_0(i_p+1, 2);
					writePart(SettingsMain, sub_folder_name, "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), *cmmPartMap[part_map_name]);
				}
				// sample velocity values from current stream function at particle positions
				else {
					std::string part_map_name = "PartA_Pos_P" + to_str_0(i_p+1, 2);
					CmmPart Part_Pos_now = *cmmPartMap[part_map_name];
					k_h_sample_points_dxdy<<<Part_Pos_now.block, Part_Pos_now.thread>>>(*Psi.Grid, *Psi.Grid, Psi.Dev_var, Part_Pos_now.Dev_var, Dev_Temp, Part_Pos_now.num);
					writeArr(SettingsMain, sub_folder_name, "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), Dev_Temp, 2*cmmPartMap[part_map_name]->num);
				}
			}
		}
	}

	return message;
}


// save one map to the map stack together with current psi
void writeMapStack(SettingsCMM SettingsMain, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Psi, int map_num, bool isForward) {
	// create new subfolder for mapstack, doesn't matter if we try to create it several times
	std::string sub_folder_name = "/MapStack";
	std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	// change name if forward or backwards map
	std::string str_forward = ""; if (isForward) str_forward = "_f";
	// change name if part of stack or last active map
	std::string map_name = to_str(map_num); if (map_num == -1) map_name = "coarse";


	// save the map
	writeTransferToBinaryFile(4*ChiX.Grid->N, ChiX.Dev_var, SettingsMain, sub_folder_name + "/Map_ChiX"  + str_forward + "_H_" + map_name, false);
	writeTransferToBinaryFile(4*ChiY.Grid->N, ChiY.Dev_var, SettingsMain, sub_folder_name + "/Map_ChiY"  + str_forward + "_H_" + map_name, false);

	// save previous psi
	// it is saved and overwritten every time but in case the simulation crashes we can then always restart from last position
	if (!isForward) {
		for (int i_psi = 1; i_psi < SettingsMain.getLagrangeOrder(); ++i_psi) {
			writeTransferToBinaryFile(4*Psi.Grid->N, Psi.Dev_var, SettingsMain, sub_folder_name + "/Stream_function_Psi_H_psi_" + to_str(i_psi), false);
		}
	}
}


void writeParticlesState(SettingsCMM SettingsMain, CmmPart** Part_Pos, CmmPart** Part_Vel) {
	// create new subfolder for mapstack, doesn't matter if we try to create it several times
	std::string sub_folder_name = "/MapStack";
	std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	// extract particle information
	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();

	// save all particle positions and velocities if tau != 0
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		// save particle positions
		writeTransferToBinaryFile(2*Part_Pos[i_p]->num, Part_Pos[i_p]->Dev_var, SettingsMain, sub_folder_name + "/Particles_advected_pos_P" + to_str_0(i_p+1, 2), false);

		// save all particle velocities if tau != 0
		if (particles_advected[i_p].tau != 0) {
			writeTransferToBinaryFile(2*Part_Vel[i_p]->num, Part_Vel[i_p]->Dev_var, SettingsMain, sub_folder_name + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), false);
		}
	}
}
std::string readParticlesState(SettingsCMM SettingsMain, CmmPart** Part_Pos, CmmPart** Part_Vel, std::string data_name) {
	// set default path
	std::string folder_name_now = data_name;
	if (data_name == "") {
		folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/MapStack";
	}
	std::string message = "";

	// extract particle information
	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();

	// read all particle positions and velocities if tau != 0
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		// read particle positions
		bool read_success = readTransferFromBinaryFile(2*Part_Pos[i_p]->num, Part_Pos[i_p]->Dev_var, folder_name_now + "/Particles_advected_pos_P" + to_str_0(i_p+1, 2) + ".data");
		if (!read_success) {
			message += "Failed to read in particles pos " + to_str(i_p) + " at location " + folder_name_now + "/Particles_advected_pos_P" + to_str_0(i_p+1, 2) + ".data\n";
		}

		// save all particle velocities if tau != 0
		if (particles_advected[i_p].tau != 0) {
			bool read_success = readTransferFromBinaryFile(2*Part_Vel[i_p]->num, Part_Vel[i_p]->Dev_var, folder_name_now + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2) + ".data");
			if (!read_success) {
				message += "Failed to read in particles vel " + to_str(i_p) + " at location " + folder_name_now + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2) + ".data\n";
			}
		}
	}

	return message;
}



Logger::Logger(SettingsCMM SettingsMain)
{
	fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/log.txt";
	if (SettingsMain.getRestartTime() == 0) file.open(fileName.c_str(), std::ios::out);
	else file.open(fileName.c_str(), std::ios::out | std::ios::app);  // append to file for continuing simulation

	if(!file)
	{
		std::cerr<<"Unable to open log file.. exitting" << std::endl;
		exit(0);
	}
	else
	{
		// new simulation, overwrite old file if existent
		if (SettingsMain.getRestartTime() == 0) file<<SettingsMain.getFileName()<< std::endl;
		// continue simulation
		else file<<"\nContinuing simulation\n"<< std::endl;
		file.close();
	}
}


void Logger::push(std::string message)
{
	file.open(fileName.c_str(), std::ios::out | std::ios::app);

	if(file)
	{
		file << "["<<currentDateTime() << "]\t";
		file << message << std::endl;
		file.close();
	}
	std::cout << message << std::endl;
}


void Logger::push()
{
	push(buffer);
}


std::string to_str_0 (int t, int width)
{
	std::ostringstream os;
	os << std::setfill('0') << std::setw(width) << t;
	return os.str ();
}

std::string hash_to_str(const void* hash, size_t size)
{
	std::ostringstream os;
    os << std::hex << std::setfill('0');
    const uint8_t* data = static_cast<const uint8_t*>(hash);
    for (size_t i = 0; i < size; ++i) {
        os << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return os.str();
}


const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

// helper function to format time to readable format
std::string format_duration(double sec) {
	return to_str(floor(sec/3600.0)) + "h " + to_str(floor(std::fmod(sec, 3600)/60.0)) + "m " + to_str(std::fmod(sec, 60)) + "s";
}

