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

#include "cmm-init.h"

#include "../grid/cmm-grid2d.h"  // for PI, twoPI and grid

#include <curand.h>
#include <curand_kernel.h>

#include "../numerical/cmm-particles.h"

__constant__ double d_rand[1000], d_init_params[10];  // array for random numbers being used for random initial conditions
extern __constant__ double d_p_init_parameter[4];  // array for particle parameters on device memory

/*******************************************************************
*			Initial condition for vorticity						   *
*******************************************************************/

__device__ double d_init_vorticity(double x, double y, int simulation_num)
{
	/*
	 *  Initial conditions for vorticity
	 *  "4_nodes" 				-	flow containing exactly 4 fourier modes with two vortices
	 *  "quadropole"			-	???
	 *  "two_vortices"			-	???
	 *  "three_vortices"		-	???
	 *  "single_shear_layer"	-	shear layer problem forming helmholtz-instabilities, merging into two vortices which then merges into one big vortex
	 *  "turbulence_gaussienne"	-	gaussian blobs - version made by thibault
	 *  "gaussian_blobs"		-	gaussian blobs in order - version made by julius
	 *  "shielded_vortex"		-	vortex core with ring of negative vorticity around it
	 */

	switch (simulation_num) {
		case 0:  // 4_nodes
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			return cos(x) + cos(y) + 0.6*cos(2*x) + 0.2*cos(3*x);
			break;
		}
		case 1:  // quadropole
		{
			double ret = 0;
			for(int iy = -2; iy <= 2; iy++)
				for(int ix = -2; ix <= 2; ix++)
				{
					double dx = x - PI/2 + ix * 2*PI;
					double dy = y - PI/2 + iy * 2*PI;
					double A = 0.6258473;
					double s = 0.5;
					double B = A/(s*s*s*s) * (dx * dy) * (dx*dx + dy*dy - 6*s*s);
					double D = (dx*dx + dy*dy)/(2*s*s);
					ret += B * exp(-D);
				}
				return ret;
			break;
		}
		case 2:  // one vortex - stationary case for investigations
		{
			double fac = 1;  // factor to increase strength
			double sigma = twoPI*0.125;  // scaling in width
			return fac / (twoPI*sigma*sigma) * exp(-((x-PI)*(x-PI) + (y-PI)*(y-PI))/(2*sigma*sigma));
			break;
		}
		case 3:  // two vortices
		{
			double ret = 0;
			for(int iy = -1; iy <= 1; iy++)
				for(int ix = -1; ix <= 1; ix++)
				{
					ret += sin(0.5*(x + twoPI*ix))*sin(0.5*(x + twoPI*ix))*sin(0.5*((y + twoPI*iy) + twoPI*iy))*sin(0.5*((y + twoPI*iy) + twoPI*iy)) * (exp(-(((x + twoPI*ix) - PI)*((x + twoPI*ix) - PI) + ((y + twoPI*iy) - 0.33*twoPI)*((y + twoPI*iy) - 0.33*twoPI))*5) +
												exp(-(((x + twoPI*ix) - PI)*((x + twoPI*ix) - PI) + ((y + twoPI*iy) - 0.67*twoPI)*((y + twoPI*iy) - 0.67*twoPI))*5));		 //two votices of same size
				}
			return ret;
			break;
		}
		case 4:  // three vortices
		{
			//three vortices
			double ret = 0;
			double LX = PI/2;
			double LY = PI/(2.0*sqrt(2.0));

			for(int iy = -1; iy <= 1; iy++)
				for(int ix = -1; ix <= 1; ix++)
				{
					ret += sin(0.5*(x + twoPI*ix))*sin(0.5*(x + twoPI*ix))*sin(0.5*((y + twoPI*iy) + twoPI*iy))*sin(0.5*((y + twoPI*iy) + twoPI*iy)) *
								(
								+	exp(-(((x + twoPI*ix) - PI - LX)*((x + twoPI*ix) - PI - LX) + ((y + twoPI*iy) - PI)*((y + twoPI*iy) - PI))*5)
								+	exp(-(((x + twoPI*ix) - PI + LX)*((x + twoPI*ix) - PI + LX) + ((y + twoPI*iy) - PI)*((y + twoPI*iy) - PI))*5)
								-	exp(-(((x + twoPI*ix) - PI + LX)*((x + twoPI*ix) - PI + LX) + ((y + twoPI*iy) - PI - LY)*((y + twoPI*iy) - PI - LY))*5)
								);		 //two votices of same size
				}
			return ret;
			break;
		}
		case 5:  // single_shear_layer
		{
			double delta = 50;  // thickness of shear layer
			double delta2 = 0.01;  // strength of instability

			return (1 + delta2*cos(2*x)) * exp(-delta*(y-PI)*(y-PI));
			break;
		}
		case 6:  // tanh_shear_layer with more parameters by julius and using a tanh velocity profile
		{
			double fac = 5;  // Factor to set freestream velocity

			double inst_strength = 0.01;  // strength of instability
			double inst_freq = 2;  // frequency of instability / how many swirls
			double shear_delta = 14 * inst_freq / twoPI;  // thickness of shear layer, physically connected, twoPi as domainsize

			// (1 + sin-perturbation) * sech^2(thickness*x)
			return fac * (1 + inst_strength * cos(inst_freq*x))  * 4 / (exp(-shear_delta*(y-PI))+exp(shear_delta*(y-PI)))
															         / (exp(-shear_delta*(y-PI))+exp(shear_delta*(y-PI)));
			break;
		}
		case 7:  // turbulence_gaussienne by thibault, similar to clercx2000 / keetels2008
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			const int NB_gaus = 8;		//NB_gaus = 6;sigma = 0.24;
			double sigma = 0.2;
			double ret = 0;

			// add positive blobs without random offset, as they sit on the corners too
			#pragma unroll
			for(int mu_x = 0; mu_x < NB_gaus; mu_x++){
				#pragma unroll
				for(int mu_y = 0; mu_y < NB_gaus; mu_y++){
					ret += 1/(twoPI*sigma*sigma)*exp(-((x-mu_x*twoPI/(NB_gaus-1))*(x-mu_x*twoPI/(NB_gaus-1))/(2*sigma*sigma)
													 + (y-mu_y*twoPI/(NB_gaus-1))*(y-mu_y*twoPI/(NB_gaus-1))/(2*sigma*sigma)));
				}
			}
			// add negative blobs with random offset
			#pragma unroll
			for(int mu_x = 0; mu_x < NB_gaus-1; mu_x++){
				#pragma unroll
				for(int mu_y = 0; mu_y < NB_gaus-1; mu_y++){
					// initialize random offset, could be changed but its working now
					curandState_t state_x;
					curand_init((mu_x+1)*mu_y*mu_y, 0, 0, &state_x);
					double RAND_gaus_x = ((double)(curand(&state_x)%1000)-500)/100000;
					curandState_t state_y;
					curand_init((mu_y+1)*mu_x*mu_x, 0, 0, &state_y);
					double RAND_gaus_y = ((double)(curand(&state_y)%1000)-500)/100000;
					ret -= 1/(twoPI*sigma*sigma)*exp(-((x-(2*mu_x+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_x)
													  *(x-(2*mu_x+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_x)/(2*sigma*sigma)
													  +(y-(2*mu_y+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_y)
													  *(y-(2*mu_y+1)*twoPI/(2*(NB_gaus-1))+RAND_gaus_y)/(2*sigma*sigma)));
				}
			}
			//curandState_t state;
			//curand_init(floor(y * 16384) * 16384 + floor(x * 16384), 0, 0, &state);
			//ret *= 1+((double)(curand(&state)%1000)-500)/100000;
			return ret - 0.008857380480028442;
			break;
		}
		case 8:  // ordered gaussienne blobs by julius, similar to clercx2000 / keetels2008
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			const int NB_gaus = 10;  // number of negative and positive blobs in each row
			double sigma = twoPI*0.01;  // scaling in width
			double fac = 1e0;  // strength scaling
			double rand_offset = twoPI*0.005;  // maximum random offset of blobs
			int border = 2;  // increase parallel domain to include values of blobs from neighbouring domains

			int dir = 1;  // a bit hackery, but it works
			double ret = 0;  // initialize value
			#pragma unroll
			for (int mu_x = -border; mu_x < 2*NB_gaus + border; mu_x++) {
				#pragma unroll
				for (int mu_y = -border; mu_y < 2*NB_gaus + border; mu_y++) {
					// comput unique number of blob, take warping into account, to have unique offset
					int blob_num = (mu_x-floor(mu_x/(double)(NB_gaus*2))*NB_gaus*2)
							     + (mu_y-floor(mu_y/(double)(NB_gaus*2))*NB_gaus*2)*NB_gaus*2;
					double exp_x = x - (mu_x+0.5)*twoPI/(2*NB_gaus) + d_rand[blob_num]*rand_offset;
					double exp_y = y - (mu_y+0.5)*twoPI/(2*NB_gaus) + d_rand[blob_num + NB_gaus*NB_gaus*4]*rand_offset;

					ret += fac * dir / (twoPI*sigma*sigma)
						 * exp(-(exp_x*exp_x + exp_y*exp_y)/(2*sigma*sigma));
					dir *= -1;
				}
				dir *= -1;  // don't alternate on line change
			}
			return ret;
			break;
		}

		// u(x,y)= - y 1/(nu^2 t^2) exp(-(x^2+y^2)/(4 nu t))
		// v(x,y)= + x 1/(nu^2 t^2) exp(-(x^2+y^2)/(4 nu t))
		case 9:  // shielded vortex
		{
			double nu = 2e-1;
			double nu_fac = 1 / (2*nu*nu*nu);  // 1 / (2*nu*nu*nu)
			double nu_center = 4*nu;  // 4*nu
			double nu_scale = 4*nu;  // 4*nu

			// compute distance from center
			double x_r = PI-x; double y_r = PI-y;

			// build vorticity
			return nu_fac * (nu_center - x_r*x_r - y_r*y_r) * exp(-(x_r*x_r + y_r*y_r)/nu_scale);
			break;
		}
		case 10:  // two_cosine as stationary case
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			return cos(x)*cos(y);
			break;
		}
		// omega = exp(-(y - phi(x))^2 / (2 delta^2)) / (sqrt(2 pi) delta^2) with phi(x) sin(x)/2
		case 11:  // vortex sheets similar to caflisch - needs one parameter which sets delta indirectly
		{
			// double Re = 5e3;
			double delta_Re = 1/sqrt(d_init_params[0]);  // rewrite for delta to reduce confusion
			// compute distance from center
			return exp(- (y-PI - sin(x)/2.0) * (y-PI - sin(x)/2.0) / (2 * delta_Re * delta_Re)) / sqrt(2*PI) / delta_Re;
		}
		case 12:  // vortex sheets with indicator function, similar to caflisch - needs two parameters: one that sets delta and one for tanh strength
		{
			// double Re = 5e3;
			double delta_Re = 1/sqrt(d_init_params[0]);  // rewrite for delta to reduce confusion
			double tanh_strength = delta_Re * d_init_params[1];  // strength of tanh curve
			// compute distance from
//			return (abs(y-PI - sin(x)/2.0) < delta_Re)  / (2.0 * delta_Re);  // sharp step function
			return (0.5*tanh((y-PI - sin(x)/2.0 + delta_Re)/tanh_strength)
				  - 0.5*tanh((y-PI - sin(x)/2.0 - delta_Re)/tanh_strength)) / (2.0 * delta_Re);  // tanh implementation
		}
		default:
			return 0;
	}

}



/*******************************************************************
*			Initial conditions for passive scalar				   *
*******************************************************************/

__device__ double d_init_scalar(double x, double y, int scalar_num) {
	/*
	 *  Initial conditions for transport of passive scalar
	 *  "rectangle"					-	simple rectangle with sharp borders
	 *  "gaussian"					-   gaussian blob
	 *  "circular_ring"				-   circular ring
	 */
	switch (scalar_num) {
		case 0:  // rectangle
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;
			double center_x = PI;  // frame center position
			double center_y = PI/2.0;  // frame center position
			double width_x = 2*PI;  // frame width
			double width_y = PI;  // frame width

			// check if position is inside of rectangle
			if (    x > center_x - width_x/2.0 and x < center_x + width_x/2.0
			    and y > center_y - width_y/2.0 and y < center_y + width_y/2.0) {
				return 1.0;
			}
			else return 0.0;
			break;
		}
		case 1:  // normal distribution / gaussian blob
		{
			double mu_x = PI;  // frame center position
			double mu_y = PI;  // frame center position
			double var_x = PI/4.0;  // variance
			double var_y = PI/4.0;  // variance

			// compute value of gaussian blob
			return 1.0/(twoPI*var_x*var_y)*exp(-((x-mu_x)*(x-mu_x)/(2*var_x*var_x)
											   + (y-mu_y)*(y-mu_y)/(2*var_y*var_y)));

			break;
		}
		case 2:  // circular ring by two tanh functions
		{
			double r = PI/2.0;  // radius of the circle
			double r_width = r/8.0;  // frame center position
			double str_1 = 10.0;  // strength of outer tanh function
			double str_2 = 10.0;  // strength of inner tanh function

			return (tanh(str_1*((x-PI)*(x-PI) + (y-PI)*(y-PI) - (r+r_width)*(r+r_width)))
				  - tanh(str_2*((x-PI)*(x-PI) + (y-PI)*(y-PI) - (r-r_width)*(r-r_width)))) / 2.0;
			break;
		}
		default:  // default - all zero
		{
			return 0.0;
			break;
		}
	}
}


/*******************************************************************
*			Initial conditions for particles					   *
*******************************************************************/

__host__ void init_particles(SettingsCMM SettingsMain, CmmPart Part, TCudaGrid2D Grid, int particle_type, int i_p) {

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

	// copy bounds and init parameters to constant memory, as we cannot pass arrays to device
	cudaMemcpyToSymbol(d_p_init_parameter, init_parameter, sizeof(double)*4);

	curandGenerator_t prng;
	// initialize randomizer
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	// set seed
	curandSetPseudoRandomGeneratorSeed(prng, seed);

	switch (init_num) {
		case 0:  // uniformly distributed in rectangle, warped to domain
		{
			// create initial positions from random distribution
			curandGenerateUniformDouble(prng, Part.Dev_var, 2*num);
			// project 0-1 onto particle frame
			k_rescale<<<Part.block, Part.thread>>>(num, Part.Dev_var, Grid);

			break;
		}
		case 1:  // normally distributed with given mean and standard deviation, warped to domain
		{
			// create all values with standard variance around 0.5, 0.5 is subtracted later in k_rescale
			curandGenerateNormalDouble( prng, Part.Dev_var, 2*num, 0.5, 1.0);
			// project 0-1 onto particle frame
			k_rescale<<<Part.block, Part.thread>>>(num, Part.Dev_var, Grid); break;
		}
		case 2:  // distributed in a circle, warped to domain
		{
			k_part_init_circle<<<Part.block, Part.thread>>>(num, Part.Dev_var, Grid); break;
		}
		case 3:  // uniform grid with equal amount of points in x- and y-direction, warped to domain
		{
			k_part_init_uniform_grid<<<Part.block, Part.thread>>>(num, Part.Dev_var, Grid); break;
		}
		case 4:  // sine distribution fitting for vortex sheets, warped to domain
		{
			k_part_init_sine_sheets<<<Part.block, Part.thread>>>(num, Part.Dev_var, Grid); break;
		}

		default:
			break;
	}
}


// kernel to compute initial positions in a circle
// d_p_init_parameter is [p_circle_center_x, p_circle_center_y, p_circle_radius_x, p_circle_radius_y]
__global__ void k_part_init_circle(int particle_num, double* Dev_particles_pos, TCudaGrid2D Grid) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	double part_pos_old[2];
	part_pos_old[0] = d_p_init_parameter[0] + d_p_init_parameter[2] * cos(i/(double)particle_num*twoPI);
	part_pos_old[1] = d_p_init_parameter[1] + d_p_init_parameter[3] * sin(i/(double)particle_num*twoPI);

	double LX = Grid.bounds[1] - Grid.bounds[0]; double LY = Grid.bounds[3] - Grid.bounds[2];
	// warping - translate by domain size(s)
	Dev_particles_pos[2*i]   = part_pos_old[0] - floor((part_pos_old[0] - Grid.bounds[0])/LX)*LX;
	Dev_particles_pos[2*i+1] = part_pos_old[1] - floor((part_pos_old[1] - Grid.bounds[2])/LY)*LY;
}


// kernel to compute initial positions in a uniform grid, points are distributed as a square for now
// d_p_init_parameter is [p_square_center_x, p_square_center_y, p_square_length_x/2, p_square_length_y/2]
__global__ void k_part_init_uniform_grid(int particle_num, double* Dev_particles_pos, TCudaGrid2D Grid) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	int length = ceil(sqrt((double)particle_num));  // distribute all points with equal points num in x- and y-direction

	double part_pos_old[2];
	part_pos_old[0] = d_p_init_parameter[0] + d_p_init_parameter[2] * (-1 + 2*(i % length)/(double)length);
	part_pos_old[1] = d_p_init_parameter[1] + d_p_init_parameter[3] * (-1 + 2*(i / length)/(double)length);

	double LX = Grid.bounds[1] - Grid.bounds[0]; double LY = Grid.bounds[3] - Grid.bounds[2];
	// warping - translate by domain size(s)
	Dev_particles_pos[2*i]   = part_pos_old[0] - floor((part_pos_old[0] - Grid.bounds[0])/LX)*LX;
	Dev_particles_pos[2*i+1] = part_pos_old[1] - floor((part_pos_old[1] - Grid.bounds[2])/LY)*LY;
}


// kernel to compute initial positions for vortex sheets center line? I'm not sure if thats whats wanted though
// d_p_init_parameter is [p_offset_x, p_offset_y, empty, empty]
__global__ void k_part_init_sine_sheets(int particle_num, double* Dev_particles_pos, TCudaGrid2D Grid) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);  // (thread_num_max * block_num + thread_num) - gives position

	// return if position is larger than particle size
	if (i >= particle_num)
		return;

	double part_pos_old[2];
	part_pos_old[0] = i/(double)particle_num*twoPI + d_p_init_parameter[0];
	part_pos_old[1] = sin(i/(double)particle_num*twoPI)/2.0 + d_p_init_parameter[1];

	double LX = Grid.bounds[1] - Grid.bounds[0]; double LY = Grid.bounds[3] - Grid.bounds[2];
	// warping - translate by domain size(s)
	Dev_particles_pos[2*i]   = part_pos_old[0] - floor((part_pos_old[0] - Grid.bounds[0])/LX)*LX;
	Dev_particles_pos[2*i+1] = part_pos_old[1] - floor((part_pos_old[1] - Grid.bounds[2])/LY)*LY;
}

/*******************************************************************
*			Initial condition for distirbution_function			   *
*******************************************************************/

__device__ double d_init_distirbution_function(double x, double v, int simulation_num)
{
	/*
	 *  Initial conditions for vorticity
	 *  "landau_damping" 		- similar to a single shear layer
	 *  "two_stream"			- similar to a double shear layer
	 */
	
	switch (simulation_num) {
		case 0:  // landau damping
		{
			double k = 0.5;  // thickness of shear layer
			double eps = 5*1e-2;  // strength of instability
			return (1 + eps*cos(k*x)) * exp(-(v)*(v)*0.5)/sqrt(2 * PI);
			break;
		}
		case 1:  // two stream
		{
			double v0 = 3;  // Factor to set freestream velocity
			double k = 0.2;  // thickness of shear layer
			double eps = 5*1e-2;  // strength of instability
			return 1/(2*sqrt(2*PI)) * (1 + eps * cos(k*x))  * (exp(-(v-v0)*(v-v0)*0.5) + exp(-(v+v0)*(v+v0)*0.5));															
			break;
		}
		default:
			return 0;
	}

}
