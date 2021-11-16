#include "cmm-io.h"


/*******************************************************************
*				     Creation of storage files					   *
*******************************************************************/

void create_directory_structure(SettingsCMM SettingsMain, double dt, int iterMax)
{
	string folder_data = SettingsMain.getWorkspace() + "data";
	struct stat st = {0};
	if (stat(folder_data.c_str(), &st) == -1) mkdir(folder_data.c_str(), 0700);

	//creating main folder
	string folder_name = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName();
	mkdir(folder_name.c_str(), 0700);

	// create general subfolder for other data
	string folder_name_tdata = folder_name + "/Monitoring_data";
	mkdir(folder_name_tdata.c_str(), 0700);

	// create general subfolder for timesteps
	folder_name_tdata = folder_name + "/Time_data";
	mkdir(folder_name_tdata.c_str(), 0700);

	// create general subfolder for zoom
	if (SettingsMain.getZoom()) {
		folder_name_tdata = folder_name + "/Zoom_data";
		mkdir(folder_name_tdata.c_str(), 0700);
	}

	string fileName = folder_name + "/readme.txt";
	ofstream file(fileName.c_str(), ios::out);

	if (!file)
	{
		cout<<"Error writting files"<<fileName<<endl;
		exit(0);
	}
	else
	{
        file<<"Simulation name \t\t\t: "<<SettingsMain.getSimName()<<endl;
        switch (SettingsMain.getTimeIntegrationNum()) {
			case 10: { file<<"Time integration \t\t\t: Euler explicit"<<endl; break; }
			case 20: { file<<"Time integration \t\t\t: Adam Bashfords 2"<<endl; break; }
			case 21: { file<<"Time integration \t\t\t: Runge Kutta 2"<<endl; break; }
			case 30: { file<<"Time integration \t\t\t: Runge Kutta 3"<<endl; break; }
			case 40: { file<<"Time integration \t\t\t: Runge Kutta 4"<<endl; break; }
			case 31: { file<<"Time integration \t\t\t: Runge Kutta 3 (modified)"<<endl; break; }
			case 41: { file<<"Time integration \t\t\t: Runge Kutta 4 (modified)"<<endl; break; }
			default: { file<<"Time integration \t\t\t: Default (zero)"<<endl; break; }
		}
        file<<"Lagrange order \t: "<<SettingsMain.getLagrangeOrder()<<endl;

        file<<"N_coarse(resolution coarse grid) \t: "<<SettingsMain.getGridCoarse()<<endl;
		file<<"N_fine(resolution fine grid) \t\t: "<<SettingsMain.getGridFine()<<endl;
		file<<"N_psi(resolution psi grid) \t\t: "<<SettingsMain.getGridPsi()<<endl;
		file<<"N_vort(resolution vort for psi grid) \t: "<<SettingsMain.getGridVort()<<endl;
		file<<"time step dt \t\t: "<<dt<<endl;
		file<<"Final time \t\t: "<<SettingsMain.getFinalTime()<<endl;
		file<<"iter max \t\t: "<<iterMax<<endl;
		file<<"Incomppressibility Threshold \t: "<<SettingsMain.getIncompThreshold()<<endl;
		file<<"Map advection epsilon \t: "<<SettingsMain.getMapEpsilon()<<endl;
		file<<"Map update order \t: "<<SettingsMain.getMapUpdateOrder()<<endl;
		file<<"Cut Psi Frequencies at \t: "<<SettingsMain.getFreqCutPsi()<<endl;
		file<<"Molly stencil version \t: "<<SettingsMain.getMollyStencil()<<endl;

		if (SettingsMain.getZoom()) {
			file<<"Zoom enabled"<<endl;
			file<<"Zoom center x : "<<SettingsMain.getZoomCenterX()<<endl;
			file<<"Zoom center y : "<<SettingsMain.getZoomCenterY()<<endl;
			file<<"Zoom width x : "<<SettingsMain.getZoomWidthX()<<endl;
			file<<"Zoom width y : "<<SettingsMain.getZoomWidthY()<<endl;
			file<<"Zoom repetitions : "<<SettingsMain.getZoomRepetitions()<<endl;
			file<<"Zoom repetition factor : "<<SettingsMain.getZoomRepetitionsFactor()<<endl;
		}

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
            if (SettingsMain.getSaveFineParticles()) {
                file<<"Safe fine Particles enabled"<<endl;
                file<<"Amount of fine particles : "<<SettingsMain.getParticlesFineNum()<<endl;
            }
        }
        else file<<"Particles disabled"<<endl;

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
        mkdir(fi.c_str(), 0700);

        // folder for fluid particle data
        string fi_1 = fi + "/Fluid";
        mkdir(fi_1.c_str(), 0700);
        // folder for fine particle data
        if (SettingsMain.getSaveFineParticles()) {
			fi_1 = fi + "/Fluid_fine";
			mkdir(fi_1.c_str(), 0700);
        }

        // folder for tau_p particles together with fine folder
        for(int i = 1; i<SettingsMain.getParticlesTauNum(); i+=1){
            fi_1 = fi + "/Tau=" + to_str(SettingsMain.particles_tau[i]);
            mkdir(fi_1.c_str(), 0700);

            if (SettingsMain.getSaveFineParticles()) {
				fi_1 = fi + "/Tau=" + to_str(SettingsMain.particles_tau[i]) + "_fine";
				mkdir(fi_1.c_str(), 0700);
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

	for (int l=0; l<Len; l++)
		file.write( (char*) &var[l], sizeof(double) );

	file.close();
}


void readAllRealFromBinaryFile(int Len, double *var, SettingsCMM SettingsMain, string data_name)
{
	string fileName = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + data_name + ".data";
	ifstream file(fileName.c_str(), ios::in | ios::binary);

		if(!file)
		{
			cout<<"Error reading file. Unable to open : "<<fileName<<endl;
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
	void writeTimeStep(SettingsCMM SettingsMain, string i_num, double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real, double *Dev_ChiX, double *Dev_ChiY, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi) {

		// create new subfolder for current timestep
		string sub_folder_name = "/Time_data/Time_" + i_num;
		string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
		struct stat st = {0};
		if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0700);

		// execute binary save for all variables
		// Vorticity on coarse grid : W_coarse
		cudaMemcpy(Host_save, Dev_W_coarse, Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Vorticity_W_coarse");
		// Vorticity on fine grid : W_fine
//		cudaMemcpy(Host_save, Dev_W_fine, Grid_fine.sizeNReal, cudaMemcpyDeviceToHost);
//	    writeAllRealToBinaryFile(Grid_fine.N, Host_save, SettingsMain, sub_folder_name + "/Vorticity_W_fine");
		// Stream function on psi grid : Psi
		cudaMemcpy(Host_save, Dev_Psi_real, 4*Grid_psi.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_psi.N, Host_save, SettingsMain, sub_folder_name + "/Stream_function_Psi_psi");
		// map in x direction on coarse grid : ChiX
		cudaMemcpy(Host_save, Dev_ChiX, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiX_coarse");
		// map in y direction on coarse grid : ChiY
		cudaMemcpy(Host_save, Dev_ChiY, 4*Grid_coarse.sizeNReal, cudaMemcpyDeviceToHost);
		writeAllRealToBinaryFile(4*Grid_coarse.N, Host_save, SettingsMain, sub_folder_name + "/Map_ChiY_coarse");
	}
#endif


// script to save only one of the variables, needed because we need temporal arrays to save
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, string i_num, double *Host_save, double *Dev_save, long int size_N, long int N) {
	// create new subfolder for current timestep, doesn't matter if we try to create it several times
	string sub_folder_name = "/Time_data/Time_" + i_num;
	string folder_name_now = SettingsMain.getWorkspace() + "data/" + SettingsMain.getFileName() + sub_folder_name;
	struct stat st = {0};
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0700);
//	mkdir(folder_name_now.c_str(), 0700);

	// copy and save
	cudaMemcpy(Host_save, Dev_save, size_N, cudaMemcpyDeviceToHost);
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
	if (stat(folder_name_now.c_str(), &st) == -1) mkdir(folder_name_now.c_str(), 0700);

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

