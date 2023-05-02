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

// header file for settings
#ifndef __CMM_PARAM_H__
#define __CMM_PARAM_H__


#include "../ui/settings.h"
#include "string.h"


// function to safe parameter file, pass-by-reference
void save_param_file(SettingsCMM& SettingsMain, std::string param_name);

// function to load parameter file, pass-by-reference
void load_param_file(SettingsCMM& SettingsMain, std::string param_name);



#endif
