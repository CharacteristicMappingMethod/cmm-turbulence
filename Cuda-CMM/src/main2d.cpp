#include "simulation/cmm-euler2d.h"
#include "ui/settings.h"
#include "ui/cmm-param.h"



/******************************************************************************************************************************
*	
*	CMM Turbulence
*
*
*   Code for the characteristic mapping method in 2D with particle flow written in C++ (C) using Nvidia CUDA on Linux.
*
*   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
*   and distribute verbatim copies of this license document, but changing it is not allowed.
*
*   Documentation and further information can be taken from the GitHub page, located at:
*   https://github.com/Arcadia197/cmm-turbulence
*
*
*	Code contributions:
*
*	Badal Yadav
*	Original version of the cuda code, developed t Mc Gill U, Montreal, Canada.
*
*	Thibault Oujia
*	Many first changes and revisions of the code
*
*	Nicolas Saber
*	Implementation of finite size particles
*
*	Julius Bergmann
*	Restructuring of the code and setup on GitHub
*
* 
* 	Literature Reference :
* 	The code is used and developed in regard to the ongoing ANR project "CM2E" (http://lmfa.ec-lyon.fr/spip.php?article1807).
* 	
* 	Badal Yadav.
* 	Characteristic mapping method for incompressible Euler equations.
* 	Master’s thesis, McGill University, Canada, 2015.
*
*   X.Y. Yin, O. Mercier, B. Yadav, K. Schneider and J.-C. Nave.
*   A Characteristic Mapping Method for the two-dimensional incompressible Euler equations.
*   J. Comput. Phys., 424, 109781, 2021.
*
*   X.Y. Yin, K. Schneider, and J.-C. Nave.
*   A characteristic mapping method for the three-dimensional incompressible Euler equations.
*   arxiv.org/abs/2107.03504, 2021b.
*
*   Nicolas Saber.
*   Two-dimensional Characteristic Mapping Method with inertial particles on GPU using CUDA.
*   Master’s thesis, Aix-Marseille University, France, 2021.
* 		
******************************************************************************************************************************/



//Main function
int main(int argc, char *args[])
{

	// build settings and apply commands
	SettingsCMM SettingsMain;

	// deal with specific arguments
	for( int count = 0; count < argc; count++ ) {
		// construct string for command
		std::string command_full = args[count];

		int pos_equal = command_full.find("=");
		if (pos_equal != std::string::npos) {
			// construct two substrings
			std::string command = command_full.substr(0, pos_equal);
			std::string value = command_full.substr(pos_equal+1, command_full.length());

			// load parameter file
			if (command == "param_file") load_param_file(SettingsMain, value);
		}
	}

	// values from command line are set after loadig parameter files - command line arguments should have highest priority
	SettingsMain.applyCommands(argc, args);

	// main function
	cuda_euler_2d(SettingsMain);

	//Zoom_load_frame("vortex_shear_1000_4", grid_scale, fine_grid_scale, "final");
	
	return 0;
}





/*******************************************************************
*					   Interesting command						   *
*******************************************************************/

/*
 * main programs location / commands for them
 *
 * compiler
 * /usr/local/cuda/bin/nvcc
 *
 * old visual profiler, sudo needed for memory investigation, reference to java needed for starting
 * good for individual profiling of functios
 * sudo /usr/local/cuda/bin/nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
 *
 * new compiler, good timeline but wrong times
 * /usr/local/cuda/bin/nsys-ui
 *
 * newest compiler, but i have to understand it
 * sudo /usr/local/cuda/bin/ncu-ui
 *
 * device memory usage information
 * nvidia-smi
 *
 * device information, very useful details
 * build:
 * cd /usr/local/cuda/samples/1_Utilities/deviceQuery
 * sudo make
 * run:
 * /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
 */


// make clean
// make CUDAFLAGS=-lineinfo SimulationCuda2d.out

// /opt/nvidia/nsight-systems/2021.1.1/bin/nsys-ui
// /opt/nvidia/nsight-compute/ncu-ui

// nsys profile --force-overwrite true -o Nsys/First_Test ./SimulationCuda2d.out
// /opt/nvidia/nsight-compute/ncu --set full -o /home/moli/Bureau/Document_Ubuntu/Code_cmm_cuda2d_Moli/Analysis/ncu_256_4096_10steps_v3 /home/moli/Bureau/Document_Ubuntu/Code_cmm_cuda2d_Moli/SimulationCuda2d.out

// compute-sanitizer  ./SimulationCuda2d.out
// cuda-gdb  ./SimulationCuda2d.out
// set cuda api_failures stop
// where



// https://docs.nvidia.com/nsight-systems/UserGuide/index.html
// https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
// https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

// https://www.youtube.com/watch?v=DnwZ6ZTLw50


// Arithmetic Interesty = Compute/Memory
// FLOPs = Floating Point Ops/second






