main: main.o kernel.o
	nvcc -o main main.o kernel.o -lcurand
main_omp: main_omp.o kernel.o
	nvcc -Xcompiler -fopenmp main_omp.o kernel.o -o main_omp -L${OPENMPI_ROOT}/lib -lmpi -lgomp -lcurand
main_mpi: main_mpi.o kernel.o
	nvcc -Xcompiler -fopenmp main_mpi.o kernel.o -o main_mpi -L${OPENMPI_ROOT}/lib -lmpi -lmpi_cxx -lpthread -lgomp -lcurand
main_cpu: main_cpu.o
	g++ -o main_cpu main_cpu.cpp
main.o: main.cpp
	nvcc -c main.o main.cpp
main_omp.o: main_omp.cpp
	nvcc -Xcompiler -fopenmp -c main_omp.o main_omp.cpp -I${OPENMPI_ROOT}/include
main_mpi.o: main_mpi.cpp
	nvcc -Xcompiler -fopenmp -c main_mpi.o main_mpi.cpp -I${OPENMPI_ROOT}/include
main_cpu.o: main_cpu.cpp
	g++ -o main_cpu main_cpu.cpp -I${OPENMPI_ROOT}/include
kernel.o: kernel.cu kernel.h
	nvcc -c kernel.o kernel.cu
clean:
	rm -f *.c *.o all
