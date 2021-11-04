# CMM Turbulence

Code for the characteristic mapping method in 2D with particle flow written in C++ (C) using Nvidia CUDA on Linux.

Even though computational resources grow rapidly, the extremely fine scales in fluid and plasma turbulence remain beyond reach using existing numerical methods. A combination of computational power and ingenious physical insights is usually needed to go beyond the brute force limit. We propose here to develop a novel numerical method, a fully Adaptive Characteristic Mapping Method (ACMM) for evolving the flow map, which yields exponential resolution in linear time. First results for the two-dimensional (2D) incompressible Euler equations show the extremely high resolution capabilities of the scheme obtained by decomposing the flow map and appropriate remapping. (ANR CM2E)

# Licensing and contributions

The code is managed under GNU General Public License v3.0. Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed. Further details can be found in the LICENSE.md file.

The original version of the Cuda code has been developed by Badal Yadav at McGill University, Montreal, Canada.
Further work on the code has been done by Thibault Oujia, Nicolas Saber and Julius Bergmann.

# Prerequesites

This code is developed to be built and run on Linux machines. In addition, an installation of Nvidia Cuda together with a featured graphics card is required. To properly install Cuda on your system, Nvidia provides a helpful guide for installation: "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html". On top of that, a working c++ implementation has to be present. The code was checked for g++ 8.4.0. The c++ code itself was develpoed under C++14.

# Compiling, building and running

1) Clone the whole repository to your local machine.
   
2) Prepare the execution of the code on your local machine:

   Checking core settings:

   - Check grid_coarse, grid_fine, grid_psi and grid_ in 'Cuda-CMM/src/simulation/settings.cu' to work with your GPU memory, the estimated memory usage while be output when running the code and can further be tested while the code is running with the command 'nvidia-smi'.
   - Change mem_RAM_CPU_remaps in 'Cuda-CMM/src/simulation/settings.cu' to match your CPU memory in MB. This value can be chosen arbitrarily high in comparison to your machines CPU RAM and will ultimately decide how many remappings can be done and saved.
   - Set the initial condition in 'Cuda-CMM/src/simulation/settings.cu'.

   Makefile:
   - set the CUDA_PATH variable to match your cuda install location, default is located under '/usr/local/cuda'
   - set the GPU_ARCH variable to match your GPU architecture. This specifies the name of the NVIDIA GPU architecture that the CUDA files will be compiled for. Further information regarding the architecture can be found at "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list". Choose a version matching your architecture name.
   - compile the code in a shell by changing into the directory and running 'make all'.
     check for errors, if not listed, please provide them as an issue along with the given error stack.
   
3) Running the code:
   - Run the code in a shell by executing 'Cuda-CMM/SimulationCuda2d.out'.
   - Variables can be set at command line level by adding pairs in the form of 'COMMAND=VALUE', the name of the commands correspond to the actual variable name in the settings file. A huge variety of information regarding the meaning and values can be found in the setPresets-function in 'Cuda-CMM/src/simulation/settings.cu'.
   - Cuda errors are quite common, any furher error however should be reported as an issue along with the error stack.

# Restrictions on code for different machines

Difference between cluster and machine computations:
The code is highly dependent on the GPU memory. This limits the usage on personal machines.

# Other notices

Known errors and bugs:
When changing many values in 'Cuda-CMM/src/simulation/settings.cu' or 'Cuda-CMM/src/simulation/settings.h', the compiler will sometimes output the message:
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
This problem can be solved by rebuilding the whole code with 'make rebuild'.

# Code structure

The main loop of the code is located in 'Cuda-CMM/src/simulation/cudaeuler2d.cu'.

Maybe some comments on what parts should be changeable and which are mainly set

Setup a Post-folder for all python post-processing?

# Literature references and Research project

The code is used and developed in regard to the ongoing ANR project "CM2E" (http://lmfa.ec-lyon.fr/spip.php?article1807).
Further literature describing the method and features:

Badal Yadav.
Characteristic mapping method for incompressible Euler equations.
Master’s thesis, McGill University, Canada, 2015.

X.Y. Yin, O. Mercier, B. Yadav, K. Schneider and J.-C. Nave. 
A Characteristic Mapping Method for the two-dimensional incompressible Euler equations. 
J. Comput. Phys., 424, 109781, 2021.

X.Y. Yin, K. Schneider, and J.-C. Nave.
A characteristic mapping method for the three-dimensional incompressible Euler equations.
arxiv.org/abs/2107.03504, 2021b.

Nicolas Saber.
Two-dimensional Characteristic Mapping Method with inertial particles on GPU using CUDA.
Master’s thesis, Aix-Marseille University, France, 2021.
