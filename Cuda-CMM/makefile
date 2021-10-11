# set the architecture after your gpu capabilities
architecture = sm_50

# main targets
SimulationCuda2d.out: settings.o cudagrid2d.o cudahermite2d.o cmm-fft.o cudaparticle2d.o cudamesure2d.o cudasimulation2d.o cudaeuler2d.o main2d.o 
	/usr/local/cuda/bin/nvcc -O3 -g -lineinfo -o SimulationCuda2d.out settings.o cudagrid2d.o cudahermite2d.o cmm-fft.o cudaparticle2d.o cudamesure2d.o cudasimulation2d.o cudaeuler2d.o main2d.o -arch=$(architecture) -lcufft -lcudadevrt -lcurand

all: settings.o cudagrid2d.o cudahermite2d.o cmm-fft.o cudaparticle2d.o cudamesure2d.o cudasimulation2d.o cudaeuler2d.o main2d.o 
	/usr/local/cuda/bin/nvcc -O3 -g -lineinfo -o SimulationCuda2d.out settings.o cudagrid2d.o cudahermite2d.o cmm-fft.o cudaparticle2d.o cudamesure2d.o cudasimulation2d.o cudaeuler2d.o main2d.o -arch=$(architecture) -lcufft -lcudadevrt -lcurand

# sub targets, non-dependent on other files
settings.o: src/simulation/settings.cu src/simulation/settings.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/simulation/settings.cu -o settings.o -arch=$(architecture) --relocatable-device-code true 
	
cudagrid2d.o: src/grid/cudagrid2d.cu src/grid/cudagrid2d.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/grid/cudagrid2d.cu -o cudagrid2d.o -arch=$(architecture) --relocatable-device-code true
	
cmm-fft.o: src/simulation/cmm-fft.cu src/simulation/cmm-fft.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/simulation/cmm-fft.cu -o cmm-fft.o -arch=$(architecture) --relocatable-device-code true 
	
cudahermite2d.o: src/hermite/cudahermite2d.cu src/hermite/cudahermite2d.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/hermite/cudahermite2d.cu -o cudahermite2d.o -arch=$(architecture) --relocatable-device-code true 

# sub targets, dependent on other files
cudaparticle2d.o: src/particle/cudaparticle2d.cu src/particle/cudaparticle2d.h src/grid/cudagrid2d.h src/hermite/cudahermite2d.h 
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/particle/cudaparticle2d.cu -o cudaparticle2d.o -arch=$(architecture) --relocatable-device-code true                                              
	
cudamesure2d.o: src/simulation/cudamesure2d.cu src/simulation/cudamesure2d.h src/grid/cudagrid2d.h src/hermite/cudahermite2d.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/simulation/cudamesure2d.cu -o cudamesure2d.o -arch=$(architecture) --relocatable-device-code true                                              
	
cudasimulation2d.o: src/simulation/cudasimulation2d.cu src/simulation/cudasimulation2d.h src/hermite/cudahermite2d.h src/grid/cudagrid2d.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/simulation/cudasimulation2d.cu -o cudasimulation2d.o -arch=$(architecture) --relocatable-device-code true
	
cudaeuler2d.o: src/simulation/cudaeuler2d.cu src/simulation/cudaeuler2d.h src/simulation/cudasimulation2d.h src/hermite/cudahermite2d.h src/grid/cudagrid2d.h
	/usr/local/cuda/bin/nvcc -O3 -c -g -lineinfo src/simulation/cudaeuler2d.cu -o cudaeuler2d.o -arch=$(architecture) --relocatable-device-code true 
	
main2d.o: src/main2d.cpp src/simulation/cudaeuler2d.h src/simulation/cudasimulation2d.h src/simulation/settings.h src/grid/cudagrid2d.h
	/usr/local/cuda/bin/nvcc -O3 -c  -g -lineinfo src/main2d.cpp -o main2d.o 
	
# rule to clean compiled files
clean:
	rm -rf *.o *.out

# Test
