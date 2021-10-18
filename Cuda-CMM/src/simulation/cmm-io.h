#ifndef __CMM_IO_H__
#define __CMM_IO_H__


// some import stuff
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "fstream"
#include "iomanip"
#include "string.h"
#include "sstream"
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>

// mkdir
#include <stdlib.h>
#include <sys/stat.h>

// programs from here
#include "../grid/cudagrid2d.h"
#include "../simulation/settings.h"

// HDF5_INCLUDE switch is implemented at compile level in the makefile
#ifdef HDF5_INCLUDE
	#include "hdf5.h"
#endif

using namespace std;

#ifdef __CUDACC__

	// general structure
	void create_directory_structure(SettingsCMM SettingsMain, string file_name, double dt, int save_buffer_count, int show_progress_at, int iterMax, int map_stack_length);
	void create_particle_directory_structure(SettingsCMM SettingsMain, string file_name, double *Tau_p, int Nb_Tau_p);

	// fundamental save functions
	void writeAllRealToBinaryFile(int Len, double *var, string workspace, string simulationName, string fileName);
	void readAllRealFromBinaryFile(int Len, double *var, string workspace, string simulationName, string fileName);

	// function to save all data from one timestep into hdf5 or other file structure
	void writeTimeStep(string workspace, string file_name, string i_num, double *Host_save, double *Dev_W_coarse, double *Dev_W_fine, double *Dev_Psi_real, double *Dev_ChiX, double *Dev_ChiY, TCudaGrid2D *Grid_fine, TCudaGrid2D *Grid_coarse, TCudaGrid2D *Grid_psi);
	void writeTimeVariable(string workspace, string sim_name, string file_name, string i_num, double *Host_save, double *Dev_save, TCudaGrid2D *Grid_save);
	void writeParticles(SettingsCMM SettingsMain, string file_name, string i_num, double *Host_particles_pos, double *Dev_particles_pos, double *Tau_p, int Nb_Tau_p);

	class Logger
	{
	public:
		Logger(string simulationName);
		void push(string message);
		void push();
		char buffer[1024];

	private:
		string fileName;
		ofstream file;
	};

	const string currentDateTime();
#endif

#endif
