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

#include "cmm-init.h"

#include "../grid/cmm-grid2d.h"  // for PI, twoPI and grid

#include <curand.h>
#include <curand_kernel.h>

#include "../numerical/cmm-particles.h"

__constant__ double d_rand[1000];  // array for random numbers being used for random initial conditions


/*******************************************************************
*			Initial conditions for particles					   *
*******************************************************************/

__host__ void init_particles(double* Dev_particles_pos, SettingsCMM SettingsMain, int particle_thread, int particle_block,
		double* domain_bounds, int particle_type, int i_p) {

	// unpack all parameters according to type
	long long int seed;
	int init_num; int num;
	double init_parameter[4];  // parameter array

	switch (particle_type) {
		case 0:  // advected particles
		{
			ParticlesAdvected* particles_advected = SettingsMain.getParticlesAdvected();
			seed = particles_advected[i_p].seed;
			init_num = particles_advected[i_p].init_num;
			num = particles_advected[i_p].num;

			init_parameter[0] = particles_advected[i_p].init_param_1;
			init_parameter[1] = particles_advected[i_p].init_param_2;
			init_parameter[2] = particles_advected[i_p].init_param_3;
			init_parameter[3] = particles_advected[i_p].init_param_4;
			break;
		}
		case 1:  // forward particles
		{
			ParticlesForwarded* particles_forwarded = SettingsMain.getParticlesForwarded();
			seed = particles_forwarded[i_p].seed;
			init_num = particles_forwarded[i_p].init_num;
			num = particles_forwarded[i_p].num;

			init_parameter[0] = particles_forwarded[i_p].init_param_1;
			init_parameter[1] = particles_forwarded[i_p].init_param_2;
			init_parameter[2] = particles_forwarded[i_p].init_param_3;
			init_parameter[3] = particles_forwarded[i_p].init_param_4;
			break;
		}
	}

	curandGenerator_t prng;
	// initialize randomizer
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	// set seed
	curandSetPseudoRandomGeneratorSeed(prng, seed);

	switch (init_num) {
		case 0:  // uniformly distributed in rectangle
		{
			// create initial positions from random distribution
			curandGenerateUniformDouble(prng, Dev_particles_pos, 2*num);

			// project 0-1 onto particle frame
			// init_param constitute to center_x, center_y, width_x, width_y
			k_rescale<<<particle_block, particle_thread>>>(num,
					init_parameter[0], init_parameter[1], init_parameter[2], init_parameter[3],
					Dev_particles_pos, domain_bounds[1], domain_bounds[3]);

			break;
		}
		case 1:  // normally distributed with given mean and standard deviation
		{
			// create all values with standard variance around 0.5, 0.5 is subtracted later in k_rescale
			curandGenerateNormalDouble( prng, Dev_particles_pos, 2*num, 0.5, 1.0);

			// shift and scale all values to fit the desired quantities
			// init_param constitute to center_x, center_y, variance_x, variance_y
			k_rescale<<<particle_block, particle_thread>>>(num,
					init_parameter[0], init_parameter[1], init_parameter[2], init_parameter[3],
					Dev_particles_pos, domain_bounds[1], domain_bounds[3]);

			break;
		}
		case 2:  // distributed in a circle
		{
			// init_param constitute to center_x, center_y, radius_x, radius_y
			k_part_init_circle<<<particle_block, particle_thread>>>(Dev_particles_pos, num,
					init_parameter[0], init_parameter[1], init_parameter[2], init_parameter[3]);
			break;
		}
		case 3:  // uniform grid with equal amount of points in x- and y-direction
		{
			// init_param constitute to center_x, center_y, length_x/2, length_y/2
			k_part_init_uniform_grid<<<particle_block, particle_thread>>>(Dev_particles_pos, num,
					init_parameter[0], init_parameter[1], init_parameter[2], init_parameter[3]);
			break;
		}
		case 4:  // sine distribution fitting for vortex sheets
		{
			// init_param constitute to offset_x, pffset_y, empty_x and empty_y
			k_part_init_sine_sheets<<<particle_block, particle_thread>>>(Dev_particles_pos, num,
					init_parameter[0], init_parameter[1], init_parameter[2], init_parameter[3]);
			break;
		}

		default:
			break;
	}
}


// kernel to compute initial positions in a circle
__global__ void k_part_init_circle(double* Dev_particles_pos, int particle_num,
		double circle_center_x, double circle_center_y, double circle_radius_x, double circle_radius_y) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	Dev_particles_pos[2*i]   = circle_center_x + circle_radius_x * cos(i/(double)particle_num*twoPI);
	Dev_particles_pos[2*i+1] = circle_center_y + circle_radius_y * sin(i/(double)particle_num*twoPI);
}


// kernel to compute initial positions in a uniform grid, points are distributed as a square for now
__global__ void k_part_init_uniform_grid(double* Dev_particles_pos, int particle_num,
		double square_center_x, double square_center_y, double square_radius_x, double square_radius_y) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	int length = ceil(sqrt((double)particle_num));  // distribute all points with equal points num in x- and y-direction

	Dev_particles_pos[2*i]   = square_center_x + square_radius_x * (-1 + 2*(i % length)/(double)length);
	Dev_particles_pos[2*i+1] = square_center_y + square_radius_y * (-1 + 2*(i / length)/(double)length);
}


// kernel to compute initial positions for vortex sheets center line? I'm not sure if thats whats wanted though
__global__ void k_part_init_sine_sheets(double* Dev_particles_pos, int particle_num,
		double offset_x, double offset_y, double empty_x, double empty_y) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	Dev_particles_pos[2*i]   = i/(double)particle_num*twoPI + offset_x;
	Dev_particles_pos[2*i+1] = sin(i/(double)particle_num*twoPI)/2.0 + offset_y;
}
