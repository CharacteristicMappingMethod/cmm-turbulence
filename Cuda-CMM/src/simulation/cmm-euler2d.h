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
