# CMM Turbulence

Code for the characteristic mapping method with particle flow written in C++ (C) using Nvidia CUDA on Linux.

Introduction here

# Licensing and contributions

The code is managed under GNU General Public License v3.0. Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed. Further details can be found in the LICENSE.md file.

The original version of the Cuda code has been developed by Badal Yadav at Mc Gill U, Montreal, Canada.

# Prerequesites

Nvidia Cuda and other packages

# Compiling, building and running

1) Copy the whole repository to your local machine.
   
2) Prepare the execution of the code on your local machine:
   
   Initialize files:
   - clone the repository to your machine, all code is located in the Cuda-CMM folder

   Checking initial conditions:
   - Check grid_scale and fine_grid_scale in Cuda-CMM/src/main2d.cpp to work with your GPU memory, the memory usage can be tested while the code is running with the command 'nvidia-smi'
   - Change mem_index in Cuda-CMM/src/simulation/cudaeuler2d.cu to a suitable number for your GPU memory. This value defines the amount of remappings to be saved on the GPU in MB and should only account for a fraction of the actual memory to give more space to other variables.
   - Change mem_ram in Cuda-CMM/src/simulation/cudaeuler2d.cu to match your CPU memory in MB. This value can be chosen arbitrarily high in comparison to your machines CPU RAM.
   - Set the initial condition in Cuda-CMM/src/main2d.cpp and Cuda-CMM/src/grid/cudagrid2d.h

   Makefile:
   - Change the -arch flag in the make-file to match your GPU architecture. This specifies the name of the NVIDIA GPU architecture that the CUDA files will be compiled for. Further information regarding the architecture can be found at "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list". Choose a version matching your architecture name.
   - check that nvcc is installed in the right directory in "/usr/local/cuda/bin/nvcc"
   - compile the code in a shell by changing into the directory and running 'make' or 'make all'
     check for errors, if not listed, please provide them as an issue along with the given error stack
   
   Running the code:
   - Run the code in a shell by executing 'SimulationCuda2d.out'
   - Cuda errors are quite common, any furher error however should be reported as an issue along with the error stack

# Restrictions on code for different machines

Difference between cluster and machine computations:
The code is highly dependent on the GPU memory. This limits the usage on personal machines.

# Other notices

Some more details, maybe team details?

# Code structure

Maybe some comments on what parts should be changeable and which are mainly set

Setup a Post-folder for all python post-processing?

# Literature References

X.Y. Yin, O. Mercier, B. Yadav, K. Schneider and J.-C. Nave. 
A Characteristic Mapping Method for the two-dimensional incompressible Euler equations. 
J. Comput. Phys., 424, 109781, 2021.
