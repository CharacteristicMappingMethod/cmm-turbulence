# CMM Turbulence

Code for the characteristic mapping method in 2D with particle flow, written in C++ (C) using Nvidia CUDA on Linux.

Even though computational resources grow rapidly, the extremely fine scales in fluid and plasma turbulence remain beyond reach using existing numerical methods. A combination of computational power and ingenious physical insights is usually needed to go beyond the brute force limit. We propose here to develop a novel numerical method, a fully Adaptive Characteristic Mapping Method (ACMM) for evolving the flow map, which yields exponential resolution in linear time. First results for the two-dimensional (2D) incompressible Euler equations show the extremely high resolution capabilities of the scheme obtained by decomposing the flow map and appropriate remapping. (ANR CM2E)

# Licensing and contributions

The code is managed under GNU General Public License v3.0. Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed. Further details can be found in the LICENSE.md file.

The original version of the Cuda code has been developed by Badal Yadav at McGill University, Montreal, Canada.
Further work on the code has been done by Thibault Oujia, Nicolas Saber and Julius Bergmann at I2M, Aix-Marseille Université, France.

# Prerequesites

This code is developed to be built and run on Linux machines. In addition, an installation of Nvidia Cuda together with a featured graphics card is required. To properly install Cuda on your system, Nvidia provides a helpful guide for installation: "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html". On top of that, a working c++ implementation has to be present. The code was checked for g++ 8.4.0. The c++ code itself was develpoed under C++14.

# Compiling, building and running

1) Clone the whole repository to your local machine.
   
2) Prepare the execution of the code on your local machine:

   Checking core settings:

   - Check grid_coarse, grid_fine, grid_psi and grid_vorticity in 'Cuda-CMM/src/simulation/settings.cu' to work with your GPU memory, the estimated memory usage is outputed usually in the console and log file when running the code and can further be tested while the code is running with the command 'nvidia-smi'.
   - Change mem_RAM_CPU_remaps in 'Cuda-CMM/src/simulation/settings.cu' to match your CPU memory in MB. This value can be chosen arbitrarily high in comparison to your machines CPU RAM and will ultimately decide how many remappings can be done and saved.
   - Set the initial condition in 'Cuda-CMM/src/simulation/settings.cu', parameter file or console input.

   Makefile:
   - set the CUDA_PATH variable to match your cuda install location, default is located under '/usr/local/cuda'
   - set the GPU_ARCH variable to match your GPU architecture. This specifies the name of the NVIDIA GPU architecture that the CUDA files will be compiled for. Further information regarding the architecture can be found at "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list". Choose a version matching your architecture name.
   - compile the code in a shell by changing into the directory and running 'make all'.
     check for errors, if not listed, please provide them as an issue along with the given error stack.
   
3) Running the code:
   - Run the code in a shell by executing the file 'Cuda-CMM/SimulationCuda2d.out'.
   - Variables can be set at command line level by adding pairs in the form of 'COMMAND=VALUE', the name of the commands correspond to the actual variable name in the settings file. A huge variety of information regarding the meaning and values can be found in the setPresets-function in 'Cuda-CMM/src/simulation/settings.cu'.
   - Further settings can be automized using parameter files. Those can be read in adding 'param_file=LOC_TO_FILE' to the console command
   - The examples folder provides a usefull guide on how to setup simulation and what settings can be adapted.
   - Cuda errors are quite common, any furher error however should be reported as an issue along with the error stack.

# Restrictions on code for different machines

The code is highly dependent on the GPU memory. This limits the usage on personal machines. a grid-size of ~1024 is suitable for testing conditions with around 1GB of GPU RAM. Further computer clusters with modern graphics card as the Nvidia Titan Tesla V100S GPU with 32GB of GPU RAM can compute sizes up to 2048 for the coarse-grid and 16384 for the fine-grid.
However, the amount of CPU RAM is also important to compute a high number of sub-maps. Using 190GB of CPU RAM can lead to 2700 sub-maps for a 1024 coarse-grid or 670 for a 2048 coarse-grid.

# Other notices

Known errors and bugs:
When changing many values in 'Cuda-CMM/src/simulation/settings.cu' or 'Cuda-CMM/src/simulation/settings.h', the compiler will sometimes output the message:

terminate called after throwing an instance of 'std::bad_alloc'

This problem can be solved by rebuilding the whole code with 'make rebuild'.

# Code structure

All source code is provided within the 'Cuda-CMM/src'-folder. This contains a folder 'Cuda-CMM/src/numerical' with all the numerical code being largely independent of the actual problem,  'Cuda-CMM/src/simulation' dealing with all CMM and Euler equation specific functions and kernels and 'Cuda-CMM/src/ui' dealing with all the file structures and user inputs.

The three most important files are 'Cuda-CMM/src/simulation/cmm-euler-2d.cu' containing the main loop of the code, 'Cuda-CMM/src/ui/settings.cu' dealing with all setting related information and 'Cuda-CMM/src/simulation/cmm-init.cu' containing all initial conditions.

# Literature references and Research project

The open access code has been developed in the context of the ongoing ANR project "CM2E" (http://lmfa.ec-lyon.fr/spip.php?article1807) and is used in this framework.
It is based on the original version of the cuda code, developed by Badal Yadav at McGill University, Canada. 
Further literature describing the method and features:

Badal Yadav. Characteristic mapping method for incompressible Euler equations.
Master thesis, McGill University, Canada, 2015.

X.Y. Yin, O. Mercier, B. Yadav, K. Schneider and J.-C. Nave. A Characteristic Mapping Method for the two-dimensional incompressible Euler equations. 
J. Comput. Phys., 424, 109781, 2021.

X.Y. Yin, K. Schneider, and J.-C. Nave. A characteristic mapping method for the three-dimensional incompressible Euler equations.
arxiv.org/abs/2107.03504, 2021b.

Nicolas Saber. Two-dimensional Characteristic Mapping Method with inertial particles on GPU using CUDA.
Master thesis, Aix-Marseille University, France, 2021.
