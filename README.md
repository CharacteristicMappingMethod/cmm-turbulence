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

2) Make sure the export path for Cuda is set by running: 
   export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
   
3) Prepare the execution of the code on your local machine:
   
   Folder structure:
   - rename the wanted src-version to 'src', this will be the working directory of the program

   Checking initial conditions:
   - Check grid_scale and fine_grid_scale in /src/main2d.cpp to work with your GPU memory
   - Change mem_index in /src/simulation/cudaeuler2d.cu to work with your GPU memory
   - Set the initial condition in /src/main2d.cpp and ??? (I forgot the second position)

   Makefile:
   - Change the -arch flag in the make-file to match your GPU architecture. This specifies the name of the NVIDIA GPU architecture that the CUDA files will be compiled for.
   - compile the code in a shell by changing into it and running 'make'
     check for further errors, if not listed, please provide them as an issue right with the given error stack
   
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
