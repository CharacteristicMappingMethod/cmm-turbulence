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

// header file for initial conditions
#ifndef __CMM_INIT_H__
#define __CMM_INIT_H__

#include "../grid/cmm-grid2d.h"  // for PI, twoPI and grid
#include "../ui/settings.h"  // for SettingsCMM

// initial condition for vorticity
__device__ double d_init_vorticity(double x, double y, int simulation_num);

// initial condition for scalars
__device__ double d_init_scalar(double x, double y, int scalar_num);

// initial condition for particles
__host__ void init_particles(double* Dev_particles_pos, SettingsCMM SettingsMain,
		int particle_thread, int particle_block, double* domain_bounds, int particle_type, int i_p);
__global__ void k_part_init_circle(double* Dev_particles_pos, int particle_num,
		double circle_center_x, double circle_center_y, double circle_radius_x, double circle_radius_y);
__global__ void k_part_init_uniform_grid(double* Dev_particles_pos, int particle_num,
		double square_center_x, double square_center_y, double square_radius_x, double square_radius_y);

#endif
