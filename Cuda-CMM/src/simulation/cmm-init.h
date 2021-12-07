// header file for initial conditions
#ifndef __CMM_INIT_H__
#define __CMM_INIT_H__

#include "../grid/cmm-grid2d.h"  // for PI, twoPI and grid

// initial condition for vorticity
__device__ double d_init_vorticity(double x, double y, int simulation_num);

// initial condition for scalars
__device__ double d_init_scalar(double x, double y, int scalar_num);

#endif
