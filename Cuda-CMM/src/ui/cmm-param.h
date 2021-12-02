// header file for settings
#ifndef __CMM_PARAM_H__
#define __CMM_PARAM_H__


#include "../ui/settings.h"
#include "string.h"


// function to safe parameter file
void save_param_file(SettingsCMM SettingsMain);

// function to load parameter file
void load_param_file(SettingsCMM SettingsMain, std::string param_name);



#endif
