goal:   main.x

tarball:
	tar cf main.tar makefile *.cpp
        
checkin:
	ci -l Makefile *.cpp

############################### pathscale ##################################  -ipo -cm -p -g test: -CB  -CA -CS -CV-ipo $(LAPACK)  
PLAT =
NVCC = nvcc
FOR = icc
NVCCFLAGS := --ptxas-options=-v -O3 -g -gencode arch=compute_20,code=sm_20 -Xptxas -v -lineinfo
FFLAGS = -O3 -cxxlib -debug -openmp -traceback
#-O3 -ipo -xHost -g -traceback
CUDADIR = /shared/ucl/apps/cuda/4.0
LIBS= -L$(CUDA_HOME)/lib64 -lcudart -lcuda -liomp5 -lcublas
INC = -I$(CUDA_HOME)/include


###############################################################################

OBJ = trove_functions.o Util.o cuda_host.o dipole_kernals.o fields.o test.o
#input.o

main.x:    main.o  $(OBJ) 
	$(FOR) -o main.x $(OBJ) $(FFLAGS) main.o $(LIBS)

main.o:       main.cu $(OBJ) 
	$(NVCC) -c main.cu $(NVCCFLAGS)  -Xcompiler -fopenmp $(INC)

trove_functions.o: trove_functions.cpp Util.o
	$(FOR) -c trove_functions.cpp $(FFLAGS)

Util.o:  Util.cpp
	$(FOR) -c Util.cpp $(FFLAGS)

fields.o:  fields.cpp
	$(FOR) -c fields.cpp $(FFLAGS)


dipole_kernals.o:  dipole_kernals.cu
	$(NVCC) -c dipole_kernals.cu $(NVCCFLAGS) 

cuda_host.o:  cuda_host.cu
	$(NVCC) -c cuda_host.cu $(NVCCFLAGS) -Xcompiler -fopenmp
test.o:  test.cu Util.o trove_functions.o
	$(NVCC) -c test.cu $(NVCCFLAGS)
#input.o:  input.f90
#       $(FOR) -c input.f90 $(FFLAGS)


clean:
	rm $(OBJ) *.o main.o




