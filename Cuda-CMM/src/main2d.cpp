#include "simulation/cudasimulation2d.h"
#include "simulation/cudaeuler2d.h"
#include "simulation/settings.h"
#include "grid/cudagrid2d.h"





/******************************************************************************************************************************
*	
* 	The original version of the Cuda code has been developed by Badal Yadav at Mc Gill U, Montreal, Canada.
* 
* 	Ref : 
* 	X.Y. Yin, O. Mercier, B. Yadav, K. Schneider and J.-C. Nave. 
* 	A Characteristic Mapping Method for the two-dimensional incompressible Euler equations. 
* 	J. Comput. Phys., 424, 109781, 2021.
* 	
* 	
* 	
* 	Main extension of the code was done by Thibault Oujia, Nicolas Saber and Julius Bergmann
* 	
* 	
* 		
******************************************************************************************************************************/



//Main function
int main(int argc, char *args[])
{

	// build settings and apply commands
	SettingsCMM SettingsMain(argc, args);

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






// sudo nvvp ./SimulationCuda2d.out


// https://docs.nvidia.com/nsight-systems/UserGuide/index.html
// https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
// https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

// https://www.youtube.com/watch?v=DnwZ6ZTLw50


// Arithmetic Interesty = Compute/Memory
// FLOPs = Floating Point Ops/second






