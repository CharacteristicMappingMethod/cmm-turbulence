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
#include <time.h>

// check for reading in directory
#include <vector>
#include <dirent.h> // for directory functions

// mkdir
#include <stdlib.h>
#include <sys/stat.h>

// programs from here
#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"

// HDF5_INCLUDE switch is implemented at compile level in the makefile
#ifdef HDF5_INCLUDE
	#include "hdf5.h"
#endif

using namespace std;

// general structure
void create_directory_structure(SettingsCMM SettingsMain, double dt, int iterMax);

// fundamental save functions
void writeAllRealToBinaryFile(int Len, double *var, SettingsCMM SettingsMain, string data_name);
void writeAppendToBinaryFile(int Len, double *var, SettingsCMM SettingsMain, string data_name);
bool readAllRealFromBinaryFile(int Len, double *var, string data_name);

// chunked copy directly from or to device memory
void writeTranferToBinaryFile(int Len, double *d_var, SettingsCMM SettingsMain, std::string data_name, bool do_append);
bool readTransferFromBinaryFile(long long int Len, double *d_var, std::string data_name);

// function to save all data from one timestep into hdf5 or other file structure
std::string writeTimeStep(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		TCudaGrid2D Grid_fine, TCudaGrid2D Grid_coarse, TCudaGrid2D Grid_psi,
		double *Dev_W_coarse, double *Dev_Psi_real,
		double *Dev_ChiX, double *Dev_ChiY, double *Dev_ChiX_f, double *Dev_ChiY_f);
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, double t_now, double *Dev_save, long int N);
void writeTimeVariable(SettingsCMM SettingsMain, string data_name, double t_now, double *Host_save, double *Dev_save, long int size_N, long int N, int offset);

std::string writeParticles(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
		double **Dev_particles_pos, double **Dev_particles_vel,
		TCudaGrid2D Grid_psi, double *Dev_psi, double *Dev_Temp, int* fluid_particles_blocks, int fluid_particles_threads);

void writeMapStack(SettingsCMM SettingsMain, MapStack Map_Stack, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi,
		double* Dev_ChiX, double* Dev_ChiY, double* Dev_W_H_fine_real, double* Dev_Psi_real, bool isForward);
void readMapStack(SettingsCMM SettingsMain, MapStack& Map_Stack, TCudaGrid2D Grid_fine, TCudaGrid2D Grid_psi,
		double* Dev_ChiX, double* Dev_ChiY, double* Dev_W_H_fine_real, double* Dev_Psi_real, bool isForward, std::string data_name);

void writeParticlesState(SettingsCMM SettingsMain, double **Dev_particles_pos, double **Dev_particles_vel);
void readParticlesState(SettingsCMM SettingsMain, double **Dev_particles_pos, double **Dev_particles_vel, std::string data_name);

class Logger
{
public:
	Logger(SettingsCMM SettingsMain);
	void push(string message);
	void push();
	char buffer[1024];

private:
	string fileName;
	ofstream file;
};

const string currentDateTime();


// little function to format datatype to string
template<typename Type> string to_str (const Type & t)
{
  ostringstream os;
  if (std::is_same<Type, double>::value) os.precision(3);
  os << t;
  return os.str ();
}
// little function to format datatype to string with specific precision
template<typename Type> string to_str (const Type & t, int prec)
{
  ostringstream os;
  os.precision(prec);
  os << t;
  return os.str ();
}
// little function to print integer with leading zeros
std::string to_str_0 (int t, int width);
// function to convert hash to hex
std::string hash_to_str(const void* hash, size_t size);

// little function to format array datatype to string
template<typename Type> string array_to_str (const Type & array, const int length)
{
  ostringstream os;
  if (std::is_same<Type, double>::value) os.precision(3);
  os << "{";
  for (int i_length = 0; i_length < length; ++i_length) {
	  os << array[i_length];
	  if (i_length != length-1) os << ",";
  }
  os << "}";
  return os.str ();
}
// little function to format array datatype to string with specific precision
template<typename Type> string array_to_str (const Type & array, const int length, int prec)
{
  ostringstream os;
  if (std::is_same<Type, double>::value) os.precision(prec);
  os << "{";
  for (int i_length = 0; i_length < length; ++i_length) {
	  os << array[i_length];
	  if (i_length != length-1) os << ",";
  }
  os << "}";
  return os.str ();
}

// little function to format milliseconds to readable format
std::string format_duration(double sec);

#endif
