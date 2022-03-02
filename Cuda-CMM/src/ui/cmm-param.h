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
