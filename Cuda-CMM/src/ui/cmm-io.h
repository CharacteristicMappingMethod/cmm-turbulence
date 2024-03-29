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
#include <cstring>
#include <time.h>
#include <dirent.h>

#include <map>  // used for a list of all variables

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

// general structure
void create_directory_structure(SettingsCMM SettingsMain, double dt, int iterMax);

// fundamental save functions
void writeAllRealToBinaryFile(size_t Len, double *var, SettingsCMM SettingsMain, std::string data_name);
void writeAppendToBinaryFile(size_t Len, double *var, SettingsCMM SettingsMain, std::string data_name);
bool readAllRealFromBinaryFile(size_t Len, double *var, std::string data_name);

// chunked copy directly from or to device memory
void writeTransferToBinaryFile(size_t Len, double *d_var, SettingsCMM SettingsMain, std::string data_name, bool do_append);
bool readTransferFromBinaryFile(size_t Len, double *d_var, std::string data_name);

// file handling
int recursive_delete(std::string path, int debug);


// function to save all data from one timestep into hdf5 or other file structure
std::string writeTimeStep(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
			std::map<std::string, CmmVar2D*> cmmVarMap, std::map<std::string, CmmPart*> cmmPartMap, double *Dev_Temp);
//std::string writeTimeStep(SettingsCMM SettingsMain, double t_now, double dt_now, double dt, std::map<std::string, CmmVar2D*> cmmVarMap);

void writeTimeVariable(SettingsCMM SettingsMain, std::string data_name, double t_now, CmmVar2D Var, size_t offset_start=0);
void writeZoomVariable(SettingsCMM SettingsMain, std::string data_name, int zoom_num, int zoom_rep, double t_now, CmmVar2D Var, size_t offset_start=0);

void writeVar(SettingsCMM SettingsMain, std::string sub_folder_name, std::string data_name, CmmVar2D Var, size_t offset_start=0, std::string debug_name_add="");
void writePart(SettingsCMM SettingsMain, std::string sub_folder_name, std::string data_name, CmmPart Part, size_t offset_start=0, std::string debug_name_add="");
void writeArr(SettingsCMM SettingsMain, std::string sub_folder_name, std::string data_name, double* Var, size_t N, size_t offset_start=0, std::string debug_name_add="");

void writeVarDebugGlobals(SettingsCMM SettingsMain, std::string data_name, CmmVar2D Var, size_t offset_start=0);
void writeDebugGlobals(SettingsCMM SettingsMain, std::string data_name, double* Var, size_t N, size_t offset_start=0);

//std::string writeParticles(SettingsCMM SettingsMain, double t_now, double dt_now, double dt,
//		std::map<std::string, CmmPart*> cmmPartMap, CmmVar2D Psi, double *Dev_Temp);

void writeMapStack(SettingsCMM SettingsMain, CmmVar2D ChiX, CmmVar2D ChiY, CmmVar2D Psi, int map_num, bool isForward);

void writeParticlesState(SettingsCMM SettingsMain, CmmPart** Part_Pos, CmmPart** Part_Vel);
std::string readParticlesState(SettingsCMM SettingsMain, CmmPart** Part_Pos, CmmPart** Part_Vel, std::string data_name);

class Logger
{
public:
	Logger(SettingsCMM SettingsMain);
	void push(std::string message);
	void push();
	char buffer[1024];

private:
	std::string fileName;
	std::ofstream file;
};

const std::string currentDateTime();


// little function to format datatype to std::string
template<typename Type> std::string to_str (const Type & t)
{
	std::ostringstream os;
	if (std::is_same<Type, double>::value) os.precision(3);
	os << t;
	return os.str ();
}
// little function to format datatype to std::string with specific precision
template<typename Type> std::string to_str (const Type & t, int prec)
{
	std::ostringstream os;
	os.precision(prec);
	os << t;
	return os.str ();
}
// little function to print integer with leading zeros
std::string to_str_0 (int t, int width);
// function to convert hash to hex
std::string hash_to_str(const void* hash, size_t size);

// little function to format array datatype to std::string
template<typename Type> std::string array_to_str (const Type & array, const int length)
{
	std::ostringstream os;
	if (std::is_same<Type, double>::value) os.precision(3);
	os << "{";
	for (int i_length = 0; i_length < length; ++i_length) {
		os << array[i_length];
		if (i_length != length-1) os << ",";
	}
	os << "}";
	return os.str ();
}
// little function to format array datatype to std::string with specific precision
template<typename Type> std::string array_to_str (const Type & array, const int length, int prec)
{
	std::ostringstream os;
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
