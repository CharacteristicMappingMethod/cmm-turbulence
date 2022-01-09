#include "cmm-init.h"

#include "../grid/cmm-grid2d.h"  // for PI, twoPI and grid

#include <curand.h>
#include <curand_kernel.h>

#include "../numerical/cmm-particles.h"

__constant__ double d_rand[1000];  // array for random numbers being used for random initial conditions

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
		case 2:  // two vortices
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
		case 3:  // three vortices
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
		case 4:  // single_shear_layer
		{
			double delta = 50;  // thickness of shear layer
			double delta2 = 0.01;  // strength of instability

			return (1 + delta2*cos(2*x)) * exp(-delta*(y-PI)*(y-PI));
			break;
		}
		case 5:  // tanh_shear_layer with more parameters by julius and using a tanh velocity profile
		{
			double fac = 5;  // factor to increase strength
			double shear_delta = 10;  // thickness of shear layer
			double inst_strength = 0.01;  // strength of instability
			double inst_freq = 4;  // frequency of instability / how many swirls

			// (1 + sin-perturbation) * sech^2(thickness*x)
			return fac * (1 + inst_strength * sin(inst_freq*x))  * 4 / (exp(-shear_delta*(y-PI))+exp(shear_delta*(y-PI)))
															         / (exp(-shear_delta*(y-PI))+exp(shear_delta*(y-PI)));
			break;
		}
		case 6:  // turbulence_gaussienne by thibault, similar to clercx2000 / keetels2008
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
		case 7:  // ordered gaussienne blobs by julius, similar to clercx2000 / keetels2008
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			const int NB_gaus = 5;  // number of negative and positive blobs in each row
			double sigma = twoPI*0.025;  // scaling in width
			double fac = 1e0;  // strength scaling
			double rand_offset = twoPI*0.0125;  // maximum random offset of blobs
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
		case 8:  // shielded vortex
		{
			double nu = 2e-1;
			double nu_fac = 1 / (2*nu*nu);  // 1 / (2*nu*nu*nu)
			double nu_center = 4*nu;  // 4*nu
			double nu_scale = 4*nu;  // 4*nu

			// compute distance from center
			double x_r = PI-x; double y_r = PI-y;

			// build vorticity
			return nu_fac * (nu_center - x_r*x_r - y_r*y_r) * exp(-(x_r*x_r + y_r*y_r)/nu_scale);
			break;
		}
		default:  //default case goes to stationary
		{
			x -= floor(x/twoPI)*twoPI;
			y -= floor(y/twoPI)*twoPI;

			return cos(x)*cos(y);
			break;
		}
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

__host__ void init_particles(double* Dev_particles_pos, SettingsCMM SettingsMain, int particle_thread, int particle_block, double* domain_bounds, int particle_type) {

	// unpack all parameters according to type
	long long int seed;
	int init_num; int num; int center_x, center_y, width_x, width_y;

	switch (particle_type) {
		case 0:  // advected particles
		{
			seed = SettingsMain.getParticlesSeed();
			init_num = SettingsMain.getParticlesInitNum();
			num = SettingsMain.getParticlesNum();
			center_x = SettingsMain.getParticlesCenterX(); center_y = SettingsMain.getParticlesCenterY();
			width_x = SettingsMain.getParticlesWidthX(); width_y = SettingsMain.getParticlesWidthY();
			break;
		}
		case 1:  // forward particles
		{
			seed = SettingsMain.getForwardParticlesSeed();
			init_num = SettingsMain.getForwardParticlesInitNum();
			num = SettingsMain.getForwardParticlesNum();
			center_x = SettingsMain.getForwardParticlesCenterX(); center_y = SettingsMain.getForwardParticlesCenterY();
			width_x = SettingsMain.getForwardParticlesWidthX(); width_y = SettingsMain.getForwardParticlesWidthY();
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
			k_rescale<<<particle_block, particle_thread>>>(num,
					center_x, center_y, width_x, width_y,
					Dev_particles_pos, domain_bounds[1], domain_bounds[3]);

			break;
		}
		case 1:  // normally distributed with given mean and standard deviation
		{
			// create all values with standard variance around 0.5, 0.5 is subtracted later in k_rescale
			curandGenerateNormalDouble( prng, Dev_particles_pos, 2*num, 0.5, 1.0);

			// shift and scale all values to fit the desired quantities, width represents the variance
			k_rescale<<<particle_block, particle_thread>>>(num,
					center_x, center_y, width_x, width_y,
					Dev_particles_pos, domain_bounds[1], domain_bounds[3]);

			break;
		}
		case 2:  // distributed in a circle
		{
			k_part_init_circle<<<particle_block, particle_thread>>>(Dev_particles_pos, num,
					center_x, center_y, width_x, width_y);
			break;
		}
		case 3:  // uniform grid
		{
			k_part_init_uniform_grid<<<particle_block, particle_thread>>>(Dev_particles_pos, num,
					center_x, center_y, width_x, width_y);
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

	int length = ceil(sqrt((double)particle_num));  // distribute all points with equal points in x- and y-direction

	Dev_particles_pos[2*i]   = square_center_x + square_radius_x * (-1 + 2*(i % length)/(double)length);
	Dev_particles_pos[2*i+1] = square_center_y + square_radius_y * (-1 + 2*(i / length)/(double)length);
}
