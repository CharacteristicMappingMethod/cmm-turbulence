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

#ifndef __CUDA_EULER_2D_H__
#define __CUDA_EULER_2D_H__

#include "../grid/cmm-grid2d.h"
#include "../ui/settings.h"


// little function being used at beginning to compute norm of velocity
struct norm_fun
{
    __host__ __device__
        double operator()(const double &x1, const double &x2) const {
            return sqrt(x1*x1 + x2*x2);
        }
};

// main function
void cuda_euler_2d(SettingsCMM& SettingsMain);


#endif
