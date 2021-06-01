#include "simulation/cudasimulation2d.h"
//#include "simulation/cudaadvection2d.h"
#include "simulation/cudaeuler2d.h"
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
* 	One sentence about the work of Thibault, Nicolas, Julius.
* 	
* 	
* 		
******************************************************************************************************************************/



// "4_nodes"		"quadropole"		"three_vortices"		"single_shear_layer"		"two_votices"
// 32		64		128		256		512		1024		2048		4096		8192		16384


//Main function
int main(int argc, char *args[])
{
	int grid_scale = 32; 
	int fine_grid_scale = 512;
	// max working on RTX 2070 : grid_scale = 2048; fine_grid_scale = 4096;
	
	cuda_euler_2d("4_nodes", grid_scale, fine_grid_scale);						//make sure to change the problem code in the cudagrid2d.h
	
	//Zoom_load_frame("vortex_shear_1000_4", grid_scale, fine_grid_scale, "final");
	
	return 0;
}





/*******************************************************************
*					   Interesting command						   *
*******************************************************************/



// nvidia-smi
// make clean
// make CUDAFLAGS=-lineinfo SimulationCuda2d.out

// /opt/nvidia/nsight-systems/2021.1.1/bin/nsys-ui
// /opt/nvidia/nsight-compute/ncu-ui

// nsys profile -o Nsys/First_Test ./SimulationCuda2d.out
// /opt/nvidia/nsight-compute/ncu --set full -o /home/moli/Bureau/Document_Ubuntu/Code_cmm_cuda2d_Moli/Analysis/ncu_256_4096_10steps_v3 /home/moli/Bureau/Document_Ubuntu/Code_cmm_cuda2d_Moli/SimulationCuda2d.out


// sudo nvvp ./SimulationCuda2d.out


// https://docs.nvidia.com/nsight-systems/UserGuide/index.html
// https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
// https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

// https://www.youtube.com/watch?v=DnwZ6ZTLw50


// Arithmetic Interesty = Compute/Memory
// FLOPs = Floating Point Ops/second



/*******************************************************************
*					 ? Interesting command ?					   *
*******************************************************************/

// --trace-fork-before-exec true

// ncu --list-sets
// ncu --set default --section SourceCounters ---metrics sm__inst_executed_pipe_tensor.sum ./my-app
// ncu --set full
// ncu --list-sections
// ncu --target-processes all









