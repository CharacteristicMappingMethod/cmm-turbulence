#include "cmm-io.h"


/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

void create_directory_structure(SettingsCMM SettingsMain, string file_name, double dt, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length)
{
	struct stat st = {0};
	if (stat("data", &st) == -1)
	{
		cout<<"A\n";
		mkdir("data", 0700);
	}

	//creating main folder
	string folder_name = SettingsMain.getWorkspace() + "data/" + file_name;
	mkdir(folder_name.c_str(), 0700);

	// create general subfolder for timesteps
	string folder_name_tdata = folder_name + "/Time_data";
	mkdir(folder_name_tdata.c_str(), 0700);

	string fileName = folder_name + "/readme.txt";
	ofstream file(fileName.c_str(), ios::out);

	if(!file)
	{
		cout<<"Error writting files"<<fileName<<endl;
		exit(0);
	}
	else
	{
        file<<"Simulation name \t\t: "<<SettingsMain.getSimName()<<endl;
        switch (SettingsMain.getTimeIntegrationNum()) {
			case 10: { file<<"Time integration : Euler explicit"<<endl; break; }
			case 20: { file<<"Time integration : Adam Bashfords 2"<<endl; break; }
			case 21: { file<<"Time integration : Runge Kutta 2"<<endl; break; }
			case 30: { file<<"Time integration : Runge Kutta 3"<<endl; break; }
			case 40: { file<<"Time integration : Runge Kutta 4"<<endl; break; }
			case 31: { file<<"Time integration : Runge Kutta 3 (modified)"<<endl; break; }
			case 41: { file<<"Time integration : Runge Kutta 4 (modified)"<<endl; break; }
			default: { file<<"Time integration : Default (zero)"<<endl; break; }
		}

        file<<"N_coarse(resolution coarse grid) \t\t: "<<SettingsMain.getGridCoarse()<<endl;
		file<<"N_fine(resolution fine grid) \t\t: "<<SettingsMain.getGridFine()<<endl;
		file<<"N_psi(resolution psi grid) \t\t: "<<SettingsMain.getGridPsi()<<endl;
		file<<"time step dt \t\t: "<<dt<<endl;
		file<<"Final time \t\t: "<<SettingsMain.getFinalTime()<<endl;
		file<<"save at \t\t: "<<save_buffer_count<<endl;
		file<<"progress at \t\t: "<<show_progress_at<<endl;
		file<<"iter max \t\t: "<<iterMax<<endl;
		file<<"stack len \t\t: "<<map_stack_length<<endl;
		file<<"Incomppressibility Threshold \t: "<<SettingsMain.getIncompThreshold()<<endl;
		file<<"Map advection epsilon \t: "<<SettingsMain.getMapEpsilon()<<endl;
		file<<"Map update order \t: "<<SettingsMain.getMapUpdateOrder()<<endl;
		if (SettingsMain.getUpsampleVersion() == 0) file<<"Psi upsample version \t : Only Psi"<<endl;
		else file<<"Psi upsample version \t : Vorticity and Psi"<<endl;
		file<<"Cut Psi Frequencies at \t:"<<SettingsMain.getFreqCutPsi()<<endl;

        if (SettingsMain.getParticles()) {
        	file<<"Particles enabled"<<endl;
        	file<<"Amount of particles : "<<SettingsMain.getParticlesNum()<<endl;
            switch (SettingsMain.getParticlesTimeIntegrationNum()) {
    			case 10: { file<<"Particles Time integration : Euler explicit"<<endl; break; }
    			case 20: { file<<"Particles Time integration : Euler midpoint"<<endl; break; }
    			case 30: { file<<"Particles Time integration : Runge Kutta 3"<<endl; break; }
    			case 40: { file<<"Particles Time integration : Runge Kutta 4"<<endl; break; }
    			case 25: { file<<"Particles Time integration : Nicolas Euler midpoint"<<endl; break; }
    			case 35: { file<<"Particles Time integration : Nicolas Runge Kutta 3"<<endl; break; }
    			default: { file<<"Particles Time integration : Default (zero)"<<endl; break; }
    		}
        }
        else file<<"Particles disabled"<<endl;

		file.close();
	}
}


// separate call for creating particle folder structure, as this is initialized later
void create_particle_directory_structure(SettingsCMM SettingsMain, string file_name, double *Tau_p, int Nb_Tau_p) {
	// create particle folders
    if (SettingsMain.getParticles()) {
    	string folder_name = SettingsMain.getWorkspace() + "data/" + file_name;

    	// main folder
        string fi = folder_name + "/Particle_data";
        mkdir(fi.c_str(), 0700);

        // folder for fluid particle data
        string fi_1 = fi + "/Fluid";
        mkdir(fi_1.c_str(), 0700);
        // folder for fine particle data
        fi_1 = fi + "/Fluid_fine";
        mkdir(fi_1.c_str(), 0700);

        // folder for tau_p particles together with fine folder
        for(int i = 1; i<Nb_Tau_p; i+=1){
            fi_1 = fi + "/Tau=" + to_str(Tau_p[i]);
            mkdir(fi_1.c_str(), 0700);

            fi_1 = fi + "/Tau=" + to_str(Tau_p[i]) + "_fine";
            mkdir(fi_1.c_str(), 0700);
        }
	}
}

/*******************************************************************
*					    Writting in binary						   *
*******************************************************************/


void writeAllRealToBinaryFile(int Len, double *var, string workspace, string simulationName, string fileName)
{
	fileName = workspace + "data/" + simulationName + fileName + ".data";
	ofstream file(fileName.c_str(), ios::out | ios::binary);

		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
			return;
		}

	for (int l=0; l<Len; l++)
		file.write( (char*) &var[l], sizeof(double) );

	file.close();
}


void readAllRealFromBinaryFile(int Len, double *var, string workspace, string simulationName, string fileName)
{
	fileName = workspace + "data/" + simulationName + fileName + ".data";
	ifstream file(fileName.c_str(), ios::in | ios::binary);

		if(!file)
		{
			cout<<"Error saving file. Unable to open : "<<fileName<<endl;
		}

	for (int l=0; l<Len; l++)
		file.read( (char*) &var[l], sizeof(double) );

	file.close();
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
	void writeTimeStep(string workspace, string file_name, string i_num, double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real, double *Dev_ChiX, double *Dev_ChiY, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi) {

	}

// binary version
#else
	void writeTimeStep(string workspace, string file_name, string i_num, double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real, double *Dev_ChiX, double *Dev_ChiY, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi) {

		// create new subfolder for current timestep
		string sub_folder_name = "/Time_data/Time_" + i_num;
		string folder_name_now = workspace + "data/" + file_name + sub_folder_name;
		mkdir(folder_name_now.c_str(), 0700);

		// execute binary save for all variables
		// Vorticity on coarse grid : W_coarse
		cudaMemcpy(Host_save, Dev_W_coarse, Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_coarse->N, Host_save, workspace, file_name, sub_folder_name + "/Vorticity_W_coarse");
		// Vorticity on fine grid : W_fine
		cudaMemcpy(Host_save, Dev_W_fine, Grid_fine->sizeNReal, cudaMemcpyDeviceToHost);
	    writeAllRealToBinaryFile(Grid_fine->N, Host_save, workspace, file_name, sub_folder_name + "/Vorticity_W_fine");
		// Stream function on psi grid : Psi
		cudaMemcpy(Host_save, Dev_Psi_real, 4*Grid_psi->sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_psi->N, Host_save, workspace, file_name, sub_folder_name + "/Stream_function_Psi_psi");
		// map in x direction on coarse grid : ChiX
		cudaMemcpy(Host_save, Dev_ChiX, 4*Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_coarse->N, Host_save, workspace, file_name, sub_folder_name + "/Map_ChiX_coarse");
		// map in y direction on coarse grid : ChiY
		cudaMemcpy(Host_save, Dev_ChiY, 4*Grid_coarse->sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_coarse->N, Host_save, workspace, file_name, sub_folder_name + "/Map_ChiY_coarse");
	}
#endif


// script to save only one of the variables, needed because we need temporal arrays to save
void writeTimeVariable(string workspace, string sim_name, string file_name, string i_num, double *Host_save, double *Dev_save, long int size_N, long int N) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	string sub_folder_name = "/Time_data/Time_" + i_num;
	string folder_name_now = workspace + "data/" + sim_name + sub_folder_name;
	mkdir(folder_name_now.c_str(), 0700);

	// copy and save
	cudaMemcpy(Host_save, Dev_save, size_N, cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(N, Host_save, workspace, sim_name, sub_folder_name + "/" + file_name);
}



/*
 * Write particle positions
 */
// will be with hdf5 version too at some point
void writeParticles(SettingsCMM SettingsMain, string file_name, string i_num, double *Host_particles_pos, double *Dev_particles_pos, double *Tau_p, int Nb_Tau_p) {
	// copy data to host
    cudaMemcpy(Host_particles_pos, Dev_particles_pos, 2*SettingsMain.getParticlesNum()*Nb_Tau_p*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    writeAllRealToBinaryFile(2*SettingsMain.getParticlesNum(), Host_particles_pos, SettingsMain.getWorkspace(), file_name, "/Particle_data/Fluid/Particles_pos_" + i_num);
    for(int i = 1; i < Nb_Tau_p; i+=1)
        writeAllRealToBinaryFile(2*SettingsMain.getParticlesNum(), Host_particles_pos + i * 2*SettingsMain.getParticlesNum(), SettingsMain.getWorkspace(), file_name, "/Particle_data/Tau=" + to_str(Tau_p[i]) + "/Particles_pos_" + i_num);
}

void writeFineParticles(SettingsCMM SettingsMain, string file_name, string i_num, double *Host_particles_fine_pos, double *Dev_particles_fine_pos, double *Tau_p, int Nb_Tau_p, int fine_particle_steps) {
	cudaMemcpy(Host_particles_fine_pos, Dev_particles_fine_pos, 2*fine_particle_steps*sizeof(double), cudaMemcpyDeviceToHost);
	writeAllRealToBinaryFile(2*fine_particle_steps, Host_particles_fine_pos, SettingsMain.getWorkspace(), file_name, "/Particle_data/Fluid_fine/Particles_pos_" + i_num);

    for(int i = 1; i < Nb_Tau_p; i+=1) {
		cudaMemcpy(Host_particles_fine_pos, Dev_particles_fine_pos + 2*i*fine_particle_steps, 2*fine_particle_steps*sizeof(double), cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(2*fine_particle_steps, Host_particles_fine_pos, SettingsMain.getWorkspace(), file_name, "/Particle_data/Tau="+to_str(Tau_p[i])+"_fine/Particles_pos_" + i_num);
    }
}


Logger::Logger(string simulationName)
{
	fileName = "data/" + simulationName + "/log.txt";
	file.open(fileName.c_str(), ios::out);

	if(!file)
	{
		cout<<"Unable to open log file.. exitting\n";
		exit(0);
	}
	else
	{
		file<<simulationName<<endl;
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

