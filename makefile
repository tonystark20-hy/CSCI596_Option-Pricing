main: main.o kernel.o
	nvcc -o main main.o kernel.o -lcurand
main_omp: main_omp.o kernel.o
    nvcc -Xcompiler -fopenmp main_omp.o kernel.o -o main_omp -I${OPENMPI_ROOT}/include -L${OPENMPI_ROOT}/lib -lmpi -lgomp -lcurand
main.o: main.cpp
	nvcc -c main.o main.cpp
main_omp.o: main_omp.cpp
    nvcc -c main_omp.o main_omp.cpp
kernel.o: kernel.cu kernel.h
	nvcc -c kernel.o kernel.cu
clean:
	rm -f *.c *.o all