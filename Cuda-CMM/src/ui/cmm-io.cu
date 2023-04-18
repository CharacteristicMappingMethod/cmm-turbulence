/******************************************************************************************************************************
*
*	This script is part of the code for the characteristic mapping method in 2D with particle flow
*	written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
******************************************************************************************************************************/

#include "cmm-io.h"

#include "../simulation/cmm-simulation-kernel.h"


/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

void create_directory_structure(SettingsCMM SettingsMain, double dt, int iterMax)
{
	string folder_data = SettingsMain.getWorkspace() + "data";
	struct stat st = {0};
	if (stat(folder_data.c_str(), &st) == -1) mkdir(folder_data.c_str(), 0777);

	//creating main folder
	string folder_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName();
	mkdir(folder_name.c_str(), 0777);

	// create general subfolder for other data
	string folder_name_tdata = folder_name + "/Monitoring_data";
	mkdir(folder_name_tdata.c_str(), 0777);

	// create general subfolder for mesure data
	folder_name_tdata = folder_name + "/Monitoring_data/Mesure/";
	mkdir(folder_name_tdata.c_str(), 0777);

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

	// empty all monitoring data so that we can later flush every value
	std::string monitoring_names[10] = {"/Error_incompressibility", "/Map_counter", "/Map_gaps", "/Time_s", "/Time_c",
			"/Mesure/Time_s", "/Mesure/Energy", "/Mesure/Enstrophy", "/Mesure/Palinstrophy", "/Mesure/Max_vorticity",
	};
	for ( const auto &i_mon_names : monitoring_names) {
		std::string fileName = folder_name + "/Monitoring_data" + i_mon_names + ".data";
		ofstream file(fileName.c_str(), std::ios::out | std::ios::trunc);
		file.close();
	}
	// empty out mesure file for sample if we do not restart
	if (SettingsMain.getRestartTime() != 0) {
		if (SettingsMain.getSaveSampleNum() > 0) {
			std::string monitoring_names[5] = {"/Time_s_", "/Energy_", "/Enstrophy_", "/Palinstrophy_", "/Max_vorticity_",
			};
			for (int i_save = 0; i_save < SettingsMain.getSaveSampleNum(); ++i_save) {
				int grid_i = SettingsMain.getSaveSample()[i_save].grid;
				std::string var_i = SettingsMain.getSaveSample()[i_save].var;
				for ( const auto &i_mon_names : monitoring_names) {
					std::string fileName = folder_name + "/Monitoring_data/Mesure" + i_mon_names + to_str(grid_i)  + ".data";
					ofstream file(fileName.c_str(), std::ios::out | std::ios::trunc);
					file.close();
				}
				if (var_i.find("Scalar") != std::string::npos or var_i.find("Theta") != std::string::npos) {
					std::string fileName = folder_name + "/Monitoring_data/Mesure/Scalar_integral_" + to_str(grid_i)  + ".data";
					ofstream file(fileName.c_str(), std::ios::out | std::ios::trunc);
					file.close();
				}
			}

		}
		if (SettingsMain.getForwardMap()) {
			std::string fileName = folder_name + "/Monitoring_data/Error_incompressibility_forward.data";
			ofstream file(fileName.c_str(), std::ios::out | std::ios::trunc);
			file.close();
			std::string fileName2 = folder_name + "/Monitoring_data/Error_invertibility.data";
			ofstream file2(fileName2.c_str(), std::ios::out | std::ios::trunc);
			file.close();
		}
	}
}


/*******************************************************************
*					    Writting in binary						   *
*******************************************************************/


void writeAllRealToBinaryFile(int Len, double *var, SettingsCMM SettingsMain, string data_name)
{
	string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	ofstream file(fileName.c_str(), ios::out | ios::binary);

	if(!file)
	{
		cout<<"Error saving file. Unable to open : "<<fileName<< std::endl;
		return;
	}
	else {
		file.write( (char*) var, Len*sizeof(double) );
	}

	file.close();
}
void writeAppendToBinaryFile(int Len, double *var, SettingsCMM SettingsMain, string data_name)
{
	string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	ofstream file(fileName.c_str(), ios::out | ios::app | ios::binary);

	if(!file)
	{
		cout<<"Error saving file. Unable to open : "<<fileName<< std::endl;
		return;
	}
	else {
		file.write( (char*) var, Len*sizeof(double) );
	}

	file.close();
}


bool readAllRealFromBinaryFile(int Len, double *var, string data_name)
{
	string fileName = data_name;
	ifstream file(fileName.c_str(), ios::in | ios::binary);
	bool open_file;

	if(!file)
	{
		cout<<"Error reading file. Unable to open : "<<fileName<< std::endl;
		open_file = false;
	}
	else {
		file.read( (char*) var, Len*sizeof(double) );
		open_file = true;
	}

	file.close();
	return open_file;
}



void writeTranferToBinaryFile(int Len, double *d_var, SettingsCMM SettingsMain, std::string data_name, bool do_append)  {
	int chunkSize = 1024;
	std::string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	std::ofstream file;
	if (!do_append) file.open(fileName.c_str(), ios::out | ios::binary);
	else file.open(fileName.c_str(), ios::out | ios::app | ios::binary);

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
		long long int offset = 0;
		while (offset < Len) {
			// check to match Len perfectly
			if (abs(Len - offset) < chunkSize) chunkSize = Len - offset;

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
bool readTransferFromBinaryFile(long long int Len, double *d_var, std::string data_name) {
	int chunkSize = 1024;
	string fileName = data_name;
	ifstream file(fileName.c_str(), ios::in | ios::binary);
	bool open_file;

	if(!file)
	{
		cout<<"Error reading file. Unable to open : "<<fileName<< std::endl;
		open_file = false;
	}
	else {
		// Create a stream
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		// Read and transfer the data in chunks
		long long int offset = 0;
		while (!file.eof() && offset < Len) {
			// check to match Len perfectly
			if (abs(Len - offset) < chunkSize) chunkSize = Len - offset;

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
			TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
			double *Dev_W_coarse, double *Dev_Psi_real,
			double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f) {

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
		std::string message = "";
		if (save_now) {
			std::string sub_folder_name;
			if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos or
					save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos or
					save_var.find("Map") != std::string::npos or save_var.find("Chi") != std::string::npos or
					save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos) {
				// create new subfolder for current timestep
				std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
				sub_folder_name = "/Time_data/Time_" + t_s_now;
				std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
				struct stat st = {0};
				if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

				// construct message here as we are sure it is being saved something
				message = "Saved computational data\n";
			}

			// Vorticity on coarse grid : W_coarse
			if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos) {
				writeTranferToBinaryFile(Grid_coarse.N, Dev_W_coarse, SettingsMain, sub_folder_name + "/Vorticity_W_coarse", false);
			}

			// Stream function on psi grid : Psi_psi
			if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
				writeTranferToBinaryFile(Grid_psi.N, Dev_Psi_real, SettingsMain, sub_folder_name + "/Stream_function_Psi_psi", false);
			}
			if (save_var.find("Stream_H") != std::string::npos or save_var.find("Psi_H") != std::string::npos) {
				writeTranferToBinaryFile(4*Grid_psi.N, Dev_Psi_real, SettingsMain, sub_folder_name + "/Stream_function_Psi_H_psi", false);
			}

			// Velocity on psi grid : U_psi
			if (save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos) {
				// Velocity in x direction
				writeTranferToBinaryFile(Grid_psi.N, Dev_Psi_real+1*Grid_psi.N, SettingsMain, sub_folder_name + "/Velocity_UX_psi", false);
				// Velocity in y direction
				writeTranferToBinaryFile(Grid_psi.N, Dev_Psi_real+2*Grid_psi.N, SettingsMain, sub_folder_name + "/Velocity_UY_psi", false);
			}

			// Backwards map on coarse grid in Hermite or single version : Chi_coarse
			if (save_var.find("Map_b") != std::string::npos or save_var.find("Chi_b") != std::string::npos) {
				// Map in x direction on coarse grid : ChiX
				writeTranferToBinaryFile(Grid_coarse.N, Dev_ChiX, SettingsMain, sub_folder_name + "/Map_ChiX_coarse", false);
				// Map in y direction on coarse grid : ChiY
				writeTranferToBinaryFile(Grid_coarse.N, Dev_ChiY, SettingsMain, sub_folder_name + "/Map_ChiY_coarse", false);
			}
			if (save_var.find("Map_H") != std::string::npos or save_var.find("Chi_H") != std::string::npos) {
				// Map in x direction on coarse grid : ChiX
				writeTranferToBinaryFile(4*Grid_coarse.N, Dev_ChiX, SettingsMain, sub_folder_name + "/Map_ChiX_H_coarse", false);
				// Map in y direction on coarse grid : ChiY
				writeTranferToBinaryFile(4*Grid_coarse.N, Dev_ChiY, SettingsMain, sub_folder_name + "/Map_ChiY_H_coarse", false);
			}

			// Forwards map on coarse grid in Hermite or single version : Chi_f_coarse
			if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
				// Map in x direction on coarse grid : ChiX
				writeTranferToBinaryFile(Grid_coarse.N, Dev_ChiX_f, SettingsMain, sub_folder_name + "/Map_ChiX_f_coarse", false);
				// Map in y direction on coarse grid : ChiY
				writeTranferToBinaryFile(Grid_coarse.N, Dev_ChiY_f, SettingsMain, sub_folder_name + "/Map_ChiY_f_coarse", false);
			}
			if (save_var.find("Map_f_H") != std::string::npos or save_var.find("Chi_f_H") != std::string::npos) {
				// Map in x direction on coarse grid : ChiX
				writeTranferToBinaryFile(4*Grid_coarse.N, Dev_ChiX_f, SettingsMain, sub_folder_name + "/Map_ChiX_f_H_coarse", false);
				// Map in y direction on coarse grid : ChiY
				writeTranferToBinaryFile(4*Grid_coarse.N, Dev_ChiY_f, SettingsMain, sub_folder_name + "/Map_ChiY_f_H_coarse", false);
			}
		}
		return message;
	}
#endif


// script to save only one of the variables, needed because we need temporal arrays to save
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, double t_now, double *Dev_save, long int N) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
	std::string sub_folder_name = "/Time_data/Time_" + t_s_now;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	// copy and save
	writeTranferToBinaryFile(N, Dev_save, SettingsMain, sub_folder_name + "/" + data_name, false);
}

// script to save only one of the variables, but with offset
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, double t_now, double *Host_save, double *Dev_save, long int size_N, long int N, int offset) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
	std::string sub_folder_name = "/Time_data/Time_" + t_s_now;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	// copy and save
	cudaMemcpy2D(Host_save, sizeof(double), Dev_save, sizeof(double)*2, sizeof(double), N, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(N, Host_save, SettingsMain, sub_folder_name + "/" + data_name);
}



/*
 * Write particle positions
 */
// will be with hdf5 version too at some point
std::string writeParticles(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, double **Dev_particles_pos, double **Dev_particles_vel,
		TCudaGrid2D Grid_psi, double *Dev_psi, double *Dev_Temp, int* fluid_particles_blocks, int fluid_particles_threads) {
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

		// create new subfolder for current timestep, doesn't matter if we try to create it several times
		if (save_var.find("PartA_", 0) != std::string::npos or save_var.find("PartAVel_", 0) != std::string::npos) {
			std::string t_s_now = to_str(t_now); if (abs(t_now - T_MAX) < 1) t_s_now = "final";
			std::string sub_folder_name = "/Particle_data/Time_" + t_s_now;
			string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
			struct stat st = {0};
			if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

			// construct message here as we are sure it is being saved something
			message = "Saved particle data\n";
		}

		for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
			// particle position first
			if (save_var.find("PartA_" + to_str_0(i_p+1, 2)) != std::string::npos) {
				writeTranferToBinaryFile(2*particles_advected[i_p].num, Dev_particles_pos[i_p], SettingsMain, "/Particle_data/Time_" + t_s_now + "/Particles_advected_pos_P" + to_str_0(i_p+1, 2), false);
			}
			// particle velocity now
			if (save_var.find("PartA_Vel_" + to_str_0(i_p+1, 2)) != std::string::npos) {
				// copy data to host and save - but only if tau != 0
				if (particles_advected[i_p].tau != 0) {
					writeTranferToBinaryFile(2*particles_advected[i_p].num, Dev_particles_vel[i_p], SettingsMain, "/Particle_data/Time_" + t_s_now + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), false);
				}
				// sample velocity values from current stream function
				else {
					k_h_sample_points_dxdy<<<fluid_particles_blocks[i_p], fluid_particles_threads>>>(Grid_psi, Grid_psi, Dev_psi, Dev_particles_pos[i_p], Dev_Temp, particles_advected[i_p].num);
					writeTranferToBinaryFile(2*particles_advected[i_p].num, Dev_Temp, SettingsMain, "/Particle_data/Time_" + t_s_now + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), false);
				}
			}
		}
	}

	return message;
}


// save the map stack together with current map and psi
void writeMapStack(SettingsCMM SettingsMain, MapStack Map_Stack, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi,
		double* Dev_ChiX, double* Dev_ChiY, double* Dev_W_H_fine_real, double* Dev_Psi_real, bool isForward) {
	// create new subfolder for mapstack, doesn't matter if we try to create it several times
	string sub_folder_name = "/MapStack";
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	string str_forward = ""; if (isForward) str_forward = "_f";

	// save every map one by one
	for (int i_map = 0; i_map < Map_Stack.map_stack_ctr; ++i_map) {
		writeAllRealToBinaryFile(4*Map_Stack.Grid->N, Map_Stack.Host_ChiX_stack_RAM[i_map], SettingsMain, sub_folder_name + "/Map_ChiX" + str_forward + "_H_" + to_str(i_map));
		writeAllRealToBinaryFile(4*Map_Stack.Grid->N, Map_Stack.Host_ChiY_stack_RAM[i_map], SettingsMain, sub_folder_name + "/Map_ChiY" + str_forward + "_H_" + to_str(i_map));
	}

	// save the active map
	writeTranferToBinaryFile(4*Map_Stack.Grid->N, Dev_ChiX, SettingsMain, sub_folder_name + "/Map_ChiX"  + str_forward + "_H_coarse", false);
	writeTranferToBinaryFile(4*Map_Stack.Grid->N, Dev_ChiY, SettingsMain, sub_folder_name + "/Map_ChiY"  + str_forward + "_H_coarse", false);

	if (!isForward) {
		// save fine vorticity Hermite
		writeTranferToBinaryFile(4*Grid_fine.N, Dev_W_H_fine_real, SettingsMain, sub_folder_name + "/Vorticity_W_H_fine", false);

		// save previous psi
		for (int i_psi = 1; i_psi < SettingsMain.getLagrangeOrder(); ++i_psi) {
			writeTranferToBinaryFile(4*Grid_psi.N, Dev_Psi_real, SettingsMain, sub_folder_name + "/Stream_function_Psi_H_psi_" + to_str(i_psi), false);
		}
	}
}


// check how many maps we can load
int countFilesWithString(const std::string& dirPath, const std::string& searchString) {
	int count = 0; DIR *dir; struct dirent *ent;

	// try to open the directory
	if ((dir = opendir (dirPath.c_str())) != NULL) {
		// loop through the directory entries
		while ((ent = readdir (dir)) != NULL) {
			// check if the entry is a regular file and if it contains the search string
			if (ent->d_type == DT_REG && std::string(ent->d_name).find(searchString) != std::string::npos) {
				count++;
			}
		}
		closedir (dir);
	} else {
		// could not open directory
		std::cerr << "Error: Could not open directory " << dirPath << std::endl;
	}
	return count;
}

// read the map stack together with current map and psi
void readMapStack(SettingsCMM SettingsMain, MapStack& Map_Stack, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi,
		double* Dev_ChiX, double* Dev_ChiY, double* Dev_W_H_fine_real, double* Dev_Psi_real, bool isForward, std::string data_name) {
	// set default path
	string folder_name_now = data_name;
	if (data_name == "") {
		folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/MapStack";
	}
	string str_forward = ""; if (isForward) str_forward = "_f";

	// read in every map one by one
	for (int i_map = 0; i_map < countFilesWithString(folder_name_now, "Map_ChiX" + str_forward + "_H") - 1; ++i_map) {
		readAllRealFromBinaryFile(4*Map_Stack.Grid->N, Map_Stack.Host_ChiX_stack_RAM[i_map], folder_name_now + "/Map_ChiX" + str_forward + "_H_" + to_str(i_map) + ".data");
		readAllRealFromBinaryFile(4*Map_Stack.Grid->N, Map_Stack.Host_ChiY_stack_RAM[i_map], folder_name_now + "/Map_ChiY" + str_forward + "_H_" + to_str(i_map) + ".data");
		Map_Stack.map_stack_ctr++;
	}

	// read the active map
	readTransferFromBinaryFile(4*Map_Stack.Grid->N, Dev_ChiX, folder_name_now + "/Map_ChiX"  + str_forward + "_H_coarse.data");
	readTransferFromBinaryFile(4*Map_Stack.Grid->N, Dev_ChiY, folder_name_now + "/Map_ChiY"  + str_forward + "_H_coarse.data");

	if (!isForward) {
		// load fine vorticity Hermite
		readTransferFromBinaryFile(4*Grid_fine.N, Dev_W_H_fine_real, folder_name_now + "/Vorticity_W_H_fine.data");

		// load previous psi
		for (int i_psi = 1; i_psi < SettingsMain.getLagrangeOrder(); ++i_psi) {
			readTransferFromBinaryFile(4*Grid_psi.N, Dev_Psi_real + i_psi*4*Grid_psi.N, folder_name_now + "/Stream_function_Psi_H_psi_" + to_str(i_psi) + ".data");
		}
	}
}


void writeParticlesState(SettingsCMM SettingsMain, double **Dev_particles_pos, double **Dev_particles_vel) {
	// create new subfolder for mapstack, doesn't matter if we try to create it several times
	string sub_folder_name = "/MapStack";
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	// extract particle information
	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();

	// save all particle positions and velocities if tau != 0
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		// save particle positions
		writeTranferToBinaryFile(2*particles_advected[i_p].num, Dev_particles_pos[i_p], SettingsMain, sub_folder_name + "/Particles_advected_pos_P" + to_str_0(i_p+1, 2), false);

		// save all particle velocities if tau != 0
		if (particles_advected[i_p].tau != 0) {
			writeTranferToBinaryFile(2*particles_advected[i_p].num, Dev_particles_vel[i_p], SettingsMain, sub_folder_name + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2), false);
		}
	}
}
void readParticlesState(SettingsCMM SettingsMain, double **Dev_particles_pos, double **Dev_particles_vel, std::string data_name) {
	// set default path
	string folder_name_now = data_name;
	if (data_name == "") {
		folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/MapStack";
	}

	// extract particle information
	ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();

	// save all particle positions and velocities if tau != 0
	for (int i_p = 0; i_p < SettingsMain.getParticlesAdvectedNum(); ++i_p) {
		// save particle positions
		readTransferFromBinaryFile(2*particles_advected[i_p].num, Dev_particles_pos[i_p], folder_name_now + "/Particles_advected_pos_P" + to_str_0(i_p+1, 2) + ".data");

		// save all particle velocities if tau != 0
		if (particles_advected[i_p].tau != 0) {
			readTransferFromBinaryFile(2*particles_advected[i_p].num, Dev_particles_vel[i_p], folder_name_now + "/Particles_advected_vel_U" + to_str_0(i_p+1, 2) + ".data");
		}
	}
}



Logger::Logger(SettingsCMM SettingsMain)
{
	fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/log.txt";
	if (SettingsMain.getRestartTime() == 0) file.open(fileName.c_str(), ios::out);
	else file.open(fileName.c_str(), ios::out | ios::app);  // append to file for continuing simulation

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


void Logger::push(string message)
{
	file.open(fileName.c_str(), ios::out | ios::app);

	if(file)
	{
		file<<"["<<currentDateTime()<<"]\t";
		file<<message<< std::endl;
		file.close();
	}
}


void Logger::push()
{
	push(buffer);
}


std::string to_str_0 (int t, int width)
{
  ostringstream os;
  os << std::setfill('0') << std::setw(width) << t;
  return os.str ();
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
string format_duration(double sec) {
	return to_str(floor(sec/3600.0)) + "h " + to_str(floor(std::fmod(sec, 3600)/60.0)) + "m " + to_str(std::fmod(sec, 60)) + "s";
}

