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
SimulationCuda2d.out: settings.o cudagrid2d.o cmm-io.o cmm-fft.o cmm-hermite.o cmm-timestep.o cmm-particles.o cudamesure2d.o cudasimulation2d.o cudaeuler2d.o main2d.o 
	#
	# Build main programm
	#
	$(NVCC_PATH) -O3 -g -lineinfo -o SimulationCuda2d.out settings.o cudagrid2d.o cmm-io.o cmm-fft.o cmm-hermite.o cmm-timestep.o cmm-particles.o cudamesure2d.o cudasimulation2d.o cudaeuler2d.o main2d.o -arch=$(GPU_ARCH) -lcufft -lcudadevrt -lcurand
	#
	# Remove all intermediate targets
	#
	# rm -rf *.o # disabled

all: SimulationCuda2d.out

# sub targets, non-dependent on other files
settings.o: src/simulation/settings.cu src/simulation/settings.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/settings.cu -o settings.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
cudagrid2d.o: src/grid/cudagrid2d.cu src/grid/cudagrid2d.h
	$(NVCC_PATH) -O3 $(HDF5_COMMAND) -c -g -lineinfo src/grid/cudagrid2d.cu -o cudagrid2d.o -arch=$(GPU_ARCH) --relocatable-device-code true

cmm-io.o: src/simulation/cmm-io.cu src/simulation/cmm-fft.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-io.cu -o cmm-io.o -arch=$(GPU_ARCH) --relocatable-device-code true 

cmm-fft.o: src/simulation/cmm-fft.cu src/simulation/cmm-fft.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cmm-fft.cu -o cmm-fft.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
cmm-hermite.o: src/numerical/cmm-hermite.cu src/numerical/cmm-hermite.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-hermite.cu -o cmm-hermite.o -arch=$(GPU_ARCH) --relocatable-device-code true 


# sub targets, dependent on other files
cmm-timestep.o: src/numerical/cmm-timestep.cu src/numerical/cmm-timestep.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-timestep.cu -o cmm-timestep.o -arch=$(GPU_ARCH) --relocatable-device-code true 

cmm-particles.o: src/numerical/cmm-particles.cu src/numerical/cmm-particles.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/numerical/cmm-particles.cu -o cmm-particles.o -arch=$(GPU_ARCH) --relocatable-device-code true                                              
	
cudamesure2d.o: src/simulation/cudamesure2d.cu src/simulation/cudamesure2d.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cudamesure2d.cu -o cudamesure2d.o -arch=$(GPU_ARCH) --relocatable-device-code true                                              
	
cudasimulation2d.o: src/simulation/cudasimulation2d.cu src/simulation/cudasimulation2d.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cudasimulation2d.cu -o cudasimulation2d.o -arch=$(GPU_ARCH) --relocatable-device-code true
	
cudaeuler2d.o: src/simulation/cudaeuler2d.cu src/simulation/cudaeuler2d.h
	$(NVCC_PATH) -O3 -c -g -lineinfo src/simulation/cudaeuler2d.cu -o cudaeuler2d.o -arch=$(GPU_ARCH) --relocatable-device-code true 
	
main2d.o: src/main2d.cpp
	$(NVCC_PATH) -O3 -c  -g -lineinfo src/main2d.cpp -o main2d.o 
	
# rule to clean compiled files
clean:
	rm -rf *.o *.out

# Test
