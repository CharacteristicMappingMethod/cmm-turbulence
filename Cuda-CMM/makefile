#	This script is part of the code for the characteristic mapping method in 2D with particle flow
#	written in C++ (C) using Nvidia CUDA on Linux.
#	
#	The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
#	and distribute verbatim copies of this license document, but changing it is not allowed.
#	
#	Documentation and further information can be taken from the GitHub page, located at:
#	https://github.com/Arcadia197/cmm-turbulence


# set cuda install path used for NVCC and thrust library
CUDA_PATH = /usr/local/cuda
NVCC_PATH = $(CUDA_PATH)/bin/nvcc
# include thrust for some algorithms
THRUST_INCLUDE = -I$(CUDA_PATH)/include/thrust -I$(CUDA_PATH)/include/cub


# set the GPU_ARCH after your gpu capabilities
# GPU_ARCH = sm_86 # for RTX 3070/ 3080/ 3090
#GPU_ARCH = sm_50  # for Julius' rusty old laptop
GPU_ARCH = compute_70

# include hdf5 or not, comment out to not include it
# HDF5_INSTALL_PATH = /usr/local/hdf5

# ifdef to include path into search file or not, not too elegant but it works
ifdef HDF5_INSTALL_PATH
	HDF5_COMMAND = -I$(HDF5_INSTALL_PATH)/include -DHDF5_INCLUDE
else
	HDF5_COMMAND = 
endif


# main targets
SimulationCuda2d.out: ./bin/globals.o ./bin/settings.o ./bin/cmm-grid2d.o ./bin/cmm-fft.o ./bin/cmm-hermite.o ./bin/cmm-io.o ./bin/cmm-param.o ./bin/cmm-timestep.o ./bin/cmm-particles.o ./bin/cmm-mesure.o ./bin/cmm-init.o ./bin/cmm-simulation-host.o ./bin/cmm-simulation-kernel.o ./bin/cmm-euler2d.o ./bin/cmm-vlasov1d.o ./bin/main2d.o
	#
	# Build main programm
	#
	$(NVCC_PATH) -O3 -g -lineinfo -o SimulationCuda2d.out ./bin/globals.o ./bin/settings.o ./bin/cmm-grid2d.o ./bin/cmm-fft.o ./bin/cmm-hermite.o ./bin/cmm-io.o ./bin/cmm-param.o ./bin/cmm-timestep.o ./bin/cmm-particles.o ./bin/cmm-mesure.o ./bin/cmm-init.o ./bin/cmm-simulation-host.o ./bin/cmm-simulation-kernel.o ./bin/cmm-euler2d.o ./bin/cmm-vlasov1d.o ./bin/main2d.o -arch=$(GPU_ARCH) -lcufft -lcudadevrt -lcurand

all: SimulationCuda2d.out

# sub targets, non-dependent on other files
./bin/globals.o: src/ui/globals.cu src/ui/globals.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/ui/globals.cu -o ./bin/globals.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/settings.o: src/ui/settings.cu src/ui/settings.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/ui/settings.cu -o ./bin/settings.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
./bin/cmm-grid2d.o: src/grid/cmm-grid2d.cu src/grid/cmm-grid2d.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/grid/cmm-grid2d.cu -o ./bin/cmm-grid2d.o -arch=$(GPU_ARCH) --relocatable-device-code true

./bin/cmm-fft.o: src/numerical/cmm-fft.cu src/numerical/cmm-fft.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-fft.cu -o ./bin/cmm-fft.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
./bin/cmm-hermite.o: src/numerical/cmm-hermite.cu src/numerical/cmm-hermite.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-hermite.cu -o ./bin/cmm-hermite.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/cmm-io.o: src/ui/cmm-io.cu src/ui/cmm-io.h | bin_dir
	$(NVCC_PATH) -O3 $(HDF5_COMMAND) -c -g -lineinfo src/ui/cmm-io.cu -o ./bin/cmm-io.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/cmm-param.o: src/ui/cmm-param.cu src/ui/cmm-param.h | bin_dir
	$(NVCC_PATH) -O3 $(HDF5_COMMAND) -c -g -lineinfo src/ui/cmm-param.cu -o ./bin/cmm-param.o -arch=$(GPU_ARCH) --relocatable-device-code true 


# sub targets, dependent on other files, but still being main numerical / computational files
./bin/cmm-timestep.o: src/numerical/cmm-timestep.cu src/numerical/cmm-timestep.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-timestep.cu -o ./bin/cmm-timestep.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/cmm-particles.o: src/numerical/cmm-particles.cu src/numerical/cmm-particles.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-particles.cu -o ./bin/cmm-particles.o -arch=$(GPU_ARCH) --relocatable-device-code true                                              
	
./bin/cmm-mesure.o: src/numerical/cmm-mesure.cu src/numerical/cmm-mesure.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-mesure.cu -o ./bin/cmm-mesure.o -arch=$(GPU_ARCH) --relocatable-device-code true                                              
	
	
# problem specific targets containing euler code
./bin/cmm-init.o: src/simulation/cmm-init.cu src/simulation/cmm-init.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-init.cu -o ./bin/cmm-init.o -arch=$(GPU_ARCH) --relocatable-device-code true


./bin/cmm-simulation-host.o: src/simulation/cmm-simulation-host.cu src/simulation/cmm-simulation-host.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-simulation-host.cu -o ./bin/cmm-simulation-host.o -arch=$(GPU_ARCH) --relocatable-device-code true

./bin/cmm-simulation-kernel.o: src/simulation/cmm-simulation-kernel.cu src/simulation/cmm-simulation-kernel.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-simulation-kernel.cu -o ./bin/cmm-simulation-kernel.o -arch=$(GPU_ARCH) --relocatable-device-code true
	
./bin/cmm-euler2d.o: src/simulation/cmm-euler2d.cu src/simulation/cmm-euler2d.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-euler2d.cu -o ./bin/cmm-euler2d.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/cmm-vlasov1d.o: src/simulation/cmm-vlasov1d.cu src/simulation/cmm-vlasov1d.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-vlasov1d.cu -o ./bin/cmm-vlasov1d.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
./bin/main2d.o: src/main2d.cpp | bin_dir
	$(NVCC_PATH) -O3 -c  -g -lineinfo src/main2d.cpp -o ./bin/main2d.o 
	
	
# rule to create bin directory
bin_dir:
	mkdir -p bin
	
# rule to clean compiled files
clean:
	rm -rf *.out
	rm -rf ./bin/*.o
	
rebuild: clean all

# Test
