# possibility to set NVCC install path to full path or just command nvcc
NVCC_PATH = /usr/local/cuda/bin/nvcc

# set the GPU_ARCH after your gpu capabilities
GPU_ARCH = sm_50

# include hdf5 or not, comment out to not include it
# HDF5_INSTALL_PATH = /usr/local/hdf5

# ifdef to include path into search file or not, not too elegant but it works
ifdef HDF5_INSTALL_PATH
	HDF5_COMMAND = -I$(HDF5_INSTALL_PATH)/include -DHDF5_INCLUDE
else
	HDF5_COMMAND = 
endif


# main targets
SimulationCuda2d.out: ./bin/settings.o ./bin/cudagrid2d.o ./bin/cmm-io.o ./bin/cmm-fft.o ./bin/cmm-hermite.o ./bin/cmm-timestep.o ./bin/cmm-particles.o ./bin/cudamesure2d.o ./bin/cudasimulation2d.o ./bin/cudaeuler2d.o ./bin/main2d.o
	#
	# Build main programm
	#
	$(NVCC_PATH) -O3 -g -lineinfo -o SimulationCuda2d.out ./bin/settings.o ./bin/cudagrid2d.o ./bin/cmm-io.o ./bin/cmm-fft.o ./bin/cmm-hermite.o ./bin/cmm-timestep.o ./bin/cmm-particles.o ./bin/cudamesure2d.o ./bin/cudasimulation2d.o ./bin/cudaeuler2d.o ./bin/main2d.o -arch=$(GPU_ARCH) -lcufft -lcudadevrt -lcurand

all: SimulationCuda2d.out

# sub targets, non-dependent on other files
./bin/settings.o: src/simulation/settings.cu src/simulation/settings.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/settings.cu -o ./bin/settings.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
./bin/cudagrid2d.o: src/grid/cudagrid2d.cu src/grid/cudagrid2d.h | bin_dir
	$(NVCC_PATH) -O3 $(HDF5_COMMAND) -c -g -lineinfo src/grid/cudagrid2d.cu -o ./bin/cudagrid2d.o -arch=$(GPU_ARCH) --relocatable-device-code true

./bin/cmm-io.o: src/ui/cmm-io.cu src/ui/cmm-io.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/ui/cmm-io.cu -o ./bin/cmm-io.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/cmm-fft.o: src/simulation/cmm-fft.cu src/simulation/cmm-fft.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-fft.cu -o ./bin/cmm-fft.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
./bin/cmm-hermite.o: src/numerical/cmm-hermite.cu src/numerical/cmm-hermite.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-hermite.cu -o ./bin/cmm-hermite.o -arch=$(GPU_ARCH) --relocatable-device-code true 


# sub targets, dependent on other files
./bin/cmm-timestep.o: src/numerical/cmm-timestep.cu src/numerical/cmm-timestep.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-timestep.cu -o ./bin/cmm-timestep.o -arch=$(GPU_ARCH) --relocatable-device-code true 

./bin/cmm-particles.o: src/numerical/cmm-particles.cu src/numerical/cmm-particles.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-particles.cu -o ./bin/cmm-particles.o -arch=$(GPU_ARCH) --relocatable-device-code true                                              
	
./bin/cudamesure2d.o: src/simulation/cudamesure2d.cu src/simulation/cudamesure2d.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cudamesure2d.cu -o ./bin/cudamesure2d.o -arch=$(GPU_ARCH) --relocatable-device-code true                                              
	
./bin/cudasimulation2d.o: src/simulation/cudasimulation2d.cu src/simulation/cudasimulation2d.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cudasimulation2d.cu -o ./bin/cudasimulation2d.o -arch=$(GPU_ARCH) --relocatable-device-code true
	
./bin/cudaeuler2d.o: src/simulation/cudaeuler2d.cu src/simulation/cudaeuler2d.h | bin_dir
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cudaeuler2d.cu -o ./bin/cudaeuler2d.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
./bin/main2d.o: src/main2d.cpp | bin_dir
	$(NVCC_PATH) -O3 -c  -g -lineinfo src/main2d.cpp -o ./bin/main2d.o 
	
	
# rule to create bin directory
bin_dir:
	mkdir -p bin
	
# rule to clean compiled files
clean:
	rm -rf *.out
	rm -rf ./bin/*.o

# Test
