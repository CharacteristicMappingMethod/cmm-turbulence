#include "cmm-io.h"


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

	// create general subfolder for timesteps
	folder_name_tdata = folder_name + "/Time_data";
	mkdir(folder_name_tdata.c_str(), 0777);

	// create general subfolder for zoom
	if (SettingsMain.getZoom()) {
		folder_name_tdata = folder_name + "/Zoom_data";
		mkdir(folder_name_tdata.c_str(), 0777);
	}

	// empty all monitoring data so that we can later flush every value
	std::string monitoring_names[7] = {"/Error_incompressibility", "/Map_counter", "/Map_gaps",
			"/Timesteps", "/Timing_Values", "/Mesure", "/Mesure_fine",
	};
	for ( const auto &i_mon_names : monitoring_names) {
		std::string fileName = folder_name + "/Monitoring_data" + i_mon_names + ".data";
		ofstream file(fileName.c_str(), std::ios::out | std::ios::trunc);
		file.close();
	}
	// empty out mesure file for sample
	if (SettingsMain.getSampleOnGrid()) {
		std::string fileName = folder_name + "/Monitoring_data/Mesure_" + to_str(SettingsMain.getGridSample()) + ".data";
		ofstream file(fileName.c_str(), std::ios::out | std::ios::trunc);
		file.close();
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


// separate call for creating particle folder structure, as this is initialized later
void create_particle_directory_structure(SettingsCMM SettingsMain) {
	// create particle folders
    if (SettingsMain.getParticles()) {
    	string folder_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName();

    	// main folder
        string fi = folder_name + "/Particle_data";
        mkdir(fi.c_str(), 0777);

        // folder for fluid particle data
        string fi_1 = fi + "/Fluid";
        mkdir(fi_1.c_str(), 0777);
        // folder for fine particle data
        if (SettingsMain.getSaveFineParticles()) {
			fi_1 = fi + "/Fluid_fine";
			mkdir(fi_1.c_str(), 0777);
        }

        // folder for tau_p particles together with fine folder
        for(int i = 1; i<SettingsMain.getParticlesTauNum(); i+=1){
            fi_1 = fi + "/Tau=" + to_str(SettingsMain.particles_tau[i]);
            mkdir(fi_1.c_str(), 0777);

            if (SettingsMain.getSaveFineParticles()) {
				fi_1 = fi + "/Tau=" + to_str(SettingsMain.particles_tau[i]) + "_fine";
				mkdir(fi_1.c_str(), 0777);
            }
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
		cout<<"Error saving file. Unable to open : "<<fileName<<endl;
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
		cout<<"Error saving file. Unable to open : "<<fileName<<endl;
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
		cout<<"Error reading file. Unable to open : "<<fileName<<endl;
		open_file = false;
	}
	else {
		file.read( (char*) var, Len*sizeof(double) );
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
void writeTimeStep(SettingsCMM SettingsMain, std::string i_num, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f) {
	}

// binary version
#else
	void writeTimeStep(SettingsCMM SettingsMain, std::string i_num, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
			double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real,
			double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f) {

		// create new subfolder for current timestep
		std::string sub_folder_name = "/Time_data/Time_" + i_num;
		std::string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
		struct stat st = {0};
		if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

		// execute binary save for all wanted variables
		std::string save_var = SettingsMain.getSaveVar();

		// Vorticity on coarse grid : W_coarse
		if (save_var.find("Vorticity") != std::string::npos or save_var.find("W") != std::string::npos) {
			cudaMemcpy(Host_save, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Vorticity_W_coarse");
		}
		// Vorticity on fine grid : W_fine
//		cudaMemcpy(Host_save, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
//	    writeAllRealToBinaryFile(Grid_fine.N, Host_save, SettingsMain, sub_folder_name + "/Vorticity_W_fine");

		// Stream function on psi grid : Psi_psi
		if (save_var.find("Stream") != std::string::npos or save_var.find("Psi") != std::string::npos) {
			cudaMemcpy(Host_save, Dev_Psi_real, Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_psi.N, Host_save, SettingsMain, sub_folder_name + "/Stream_function_Psi_psi");
		}
		if (save_var.find("Stream_H") != std::string::npos or save_var.find("Psi_H") != std::string::npos) {
			cudaMemcpy(Host_save, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(4*Grid_psi.N, Host_save, SettingsMain, sub_folder_name + "/Stream_function_Psi_H_psi");
		}

		// Velocity on psi grid : U_psi
		if (save_var.find("Velocity") != std::string::npos or save_var.find("U") != std::string::npos) {
			// Velocity in x direction
			cudaMemcpy(Host_save, Dev_Psi_real+1*Grid_psi.N, Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_psi.N, Host_save, SettingsMain, sub_folder_name + "/Velocity_UX_psi");
			// Velocity in y direction
			cudaMemcpy(Host_save, Dev_Psi_real+2*Grid_psi.N, Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_psi.N, Host_save, SettingsMain, sub_folder_name + "/Velocity_UY_psi");
		}

		// Backwards map on coarse grid in Hermite or single version : Chi_coarse
		if (save_var.find("Map") != std::string::npos or save_var.find("Chi") != std::string::npos) {
			// Map in x direction on coarse grid : ChiX
			cudaMemcpy(Host_save, Dev_ChiX, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiX_coarse");
			// Map in y direction on coarse grid : ChiY
			cudaMemcpy(Host_save, Dev_ChiY, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiY_coarse");
		}
		if (save_var.find("Map_H") != std::string::npos or save_var.find("Chi_H") != std::string::npos) {
			// Map in x direction on coarse grid : ChiX
			cudaMemcpy(Host_save, Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(4*Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiX_H_coarse");
			// Map in y direction on coarse grid : ChiY
			cudaMemcpy(Host_save, Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(4*Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiY_H_coarse");
		}

		// Forwards map on coarse grid in Hermite or single version : Chi_f_coarse
		if (save_var.find("Map_f") != std::string::npos or save_var.find("Chi_f") != std::string::npos) {
			// Map in x direction on coarse grid : ChiX
			cudaMemcpy(Host_save, Dev_ChiX_f, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiX_f_coarse");
			// Map in y direction on coarse grid : ChiY
			cudaMemcpy(Host_save, Dev_ChiY_f, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiY_f_coarse");
		}
		if (save_var.find("Map_f_H") != std::string::npos or save_var.find("Chi_f_H") != std::string::npos) {
			// Map in x direction on coarse grid : ChiX
			cudaMemcpy(Host_save, Dev_ChiX_f, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(4*Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiX_f_H_coarse");
			// Map in y direction on coarse grid : ChiY
			cudaMemcpy(Host_save, Dev_ChiY_f, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
			writeAllRealToBinaryFile(4*Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiY_f_H_coarse");
		}
	}
#endif


// script to save only one of the variables, needed because we need temporal arrays to save
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, string i_num, double *Host_save, double *Dev_save, long int size_N, long int N) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	string sub_folder_name = "/Time_data/Time_" + i_num;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);
//	mkdir(folder_name_now.c_str(), 0777);

	// copy and save
	cudaMemcpy(Host_save, Dev_save, size_N, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(N, Host_save, SettingsMain, sub_folder_name + "/" + data_name);
}

// script to save only one of the variables, but with offset
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, string i_num, double *Host_save, double *Dev_save, long int size_N, long int N, int offset) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	string sub_folder_name = "/Time_data/Time_" + i_num;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);
//	mkdir(folder_name_now.c_str(), 0777);

	// copy and save
	cudaMemcpy2D(Host_save, sizeof(double), Dev_save, sizeof(double)*2,
			sizeof(double), N, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(N, Host_save, SettingsMain, sub_folder_name + "/" + data_name);
}



/*
 * Write particle positions
 */
// will be with hdf5 version too at some point
void writeParticles(SettingsCMM SettingsMain, string i_num, double *Host_particles_pos, double *Dev_particles_pos) {
	// copy data to host
    cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*SettingsMain.getParticlesNum()*SettingsMain.getParticlesTauNum()*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    writeAllRealToBinaryFile(2*SettingsMain.getParticlesNum(), Host_particles_pos, SettingsMain, "/Particle_data/Fluid/Particles_pos_" + i_num);
    for(int i = 1; i < SettingsMain.getParticlesTauNum(); i+=1)
        writeAllRealToBinaryFile(2*SettingsMain.getParticlesNum(), Host_particles_pos + i * 2*SettingsMain.getParticlesNum(), SettingsMain, "/Particle_data/Tau=" + to_str(SettingsMain.particles_tau[i]) + "/Particles_pos_" + i_num);
}

void writeFineParticles(SettingsCMM SettingsMain, string i_num, double *Host_particles_fine_pos, int fine_particle_save_num) {
	writeAllRealToBinaryFile(2*fine_particle_save_num, Host_particles_fine_pos, SettingsMain, "/Particle_data/Fluid_fine/Particles_pos_" + i_num);

    for(int i = 1; i < SettingsMain.getParticlesTauNum(); i+=1) {
		writeAllRealToBinaryFile(2*fine_particle_save_num, Host_particles_fine_pos, SettingsMain, "/Particle_data/Tau="+to_str(SettingsMain.particles_tau[i])+"_fine/Particles_pos_" + i_num);
    }
}


// save the map stack, only save used maps though
void writeMapStack(SettingsCMM SettingsMain, MapStack Map_Stack) {
	// create new subfolder for mapstack, doesn't matter if we try to create it several times
	string sub_folder_name = "/MapStack";
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0777);

	// check if we have to save a stack for every stack
	int save_ctr;
	if (Map_Stack.map_stack_ctr / (double)Map_Stack.cpu_map_num > 0) {
		if (Map_Stack.map_stack_ctr > 1*Map_Stack.cpu_map_num) save_ctr = Map_Stack.cpu_map_num;
		else save_ctr = Map_Stack.map_stack_ctr - 0*Map_Stack.cpu_map_num;
		printf("Save %d maps of map stack 1\n",save_ctr);
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiX_stack_RAM_0, SettingsMain, "/MapStack/MapStack_ChiX_0");
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiY_stack_RAM_0, SettingsMain, "/MapStack/MapStack_ChiY_0");
	}
	else if (Map_Stack.map_stack_ctr / (double)Map_Stack.cpu_map_num > 1) {
		if (Map_Stack.map_stack_ctr > 2*Map_Stack.cpu_map_num) save_ctr = Map_Stack.cpu_map_num;
		else save_ctr = Map_Stack.map_stack_ctr - 1*Map_Stack.cpu_map_num;
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiX_stack_RAM_1, SettingsMain, "/MapStack/MapStack_ChiX_1");
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiY_stack_RAM_1, SettingsMain, "/MapStack/MapStack_ChiY_1");
	}
	else if (Map_Stack.map_stack_ctr / (double)Map_Stack.cpu_map_num > 2) {
		if (Map_Stack.map_stack_ctr > 3*Map_Stack.cpu_map_num) save_ctr = Map_Stack.cpu_map_num;
		else save_ctr = Map_Stack.map_stack_ctr - 2*Map_Stack.cpu_map_num;
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiX_stack_RAM_2, SettingsMain, "/MapStack/MapStack_ChiX_2");
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiY_stack_RAM_2, SettingsMain, "/MapStack/MapStack_ChiY_2");
	}
	else if (Map_Stack.map_stack_ctr / (double)Map_Stack.cpu_map_num > 3) {
		if (Map_Stack.map_stack_ctr > 4*Map_Stack.cpu_map_num) save_ctr = Map_Stack.cpu_map_num;
		else save_ctr = Map_Stack.map_stack_ctr - 3*Map_Stack.cpu_map_num;
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiX_stack_RAM_3, SettingsMain, "/MapStack/MapStack_ChiX_3");
		writeAllRealToBinaryFile(save_ctr*4*Map_Stack.Grid->N, Map_Stack.Host_ChiY_stack_RAM_3, SettingsMain, "/MapStack/MapStack_ChiY_3");
	}
}


Logger::Logger(SettingsCMM SettingsMain)
{
	fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + "/log.txt";
	file.open(fileName.c_str(), ios::out);
//	file.open(fileName.c_str(), ios::out | ios::app);  // append to file for continuing simulation

	if(!file)
	{
		cout<<"Unable to open log file.. exitting\n";
		exit(0);
	}
	else
	{
		file<<SettingsMain.getFileName()<<endl;  // if file existed, this will basically overwrite it
		file.close();
	}
}


void Logger::push(string message)
{
	file.open(fileName.c_str(), ios::out | ios::app);

	if(file)
	{
		file<<"["<<currentDateTime()<<"]\t";
		file<<message<<endl;
		file.close();
	}
}


void Logger::push()
{
	push(buffer);
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

