SimulationCuda2d.out: cudagrid2d.o cudahermite2d.o cudasimulation2d.o cudaeuler2d.o main2d.o 
	nvcc -O3 -o SimulationCuda2d.out cudagrid2d.o cudahermite2d.o cudasimulation2d.o cudaeuler2d.o main2d.o -arch=sm_70 -lcufft -lcudadevrt
	
cudagrid2d.o: src/grid/cudagrid2d.cu src/grid/cudagrid2d.h
	nvcc -O3 -c src/grid/cudagrid2d.cu -o cudagrid2d.o -arch=sm_70 --relocatable-device-code true
	
cudahermite2d.o: src/hermite/cudahermite2d.cu src/hermite/cudahermite2d.h src/grid/cudagrid2d.h 
	nvcc -O3 -c src/hermite/cudahermite2d.cu -o cudahermite2d.o -arch=sm_70 --relocatable-device-code true                                               
	
cudasimulation2d.o: src/simulation/cudasimulation2d.cu src/simulation/cudasimulation2d.h src/hermite/cudahermite2d.h src/grid/cudagrid2d.h
	nvcc -O3 -c src/simulation/cudasimulation2d.cu -o cudasimulation2d.o -arch=sm_70 --relocatable-device-code true
	
cudaeuler2d.o: src/simulation/cudaeuler2d.cu src/simulation/cudaeuler2d.h src/simulation/cudasimulation2d.h src/hermite/cudahermite2d.h src/grid/cudagrid2d.h
	nvcc -O3 -c src/simulation/cudaeuler2d.cu -o cudaeuler2d.o -arch=sm_70 --relocatable-device-code true
	
main2d.o: src/main2d.cpp src/simulation/cudaeuler2d.h src/simulation/cudasimulation2d.h src/grid/cudagrid2d.h
	nvcc -O3 -c src/main2d.cpp -o main2d.o 
	
clean:
	rm -rf *.o *.out
