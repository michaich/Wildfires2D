bs ?= 8 #each block has bs x bs grid points (default here is 8 x 8)
openmp ?= false

# Set compiler
ifneq ($(MPICXX),)
  CXX=$(MPICXX)
else
  CXX=mpic++
endif

# Set compiler flags
CPPFLAGS= -std=c++17 -g

# Set flags for warnings (gcc compiler needed)
CPPFLAGS+= -Wall -Wextra -Wfloat-equal -Woverloaded-virtual -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings -Wno-unknown-pragmas -Wno-unused-parameter -Wno-deprecated-declarations -Wno-unused-result -Wno-cast-function-type

# Set flags for compiler optimizations
CPPFLAGS+= -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native -falign-functions -ftree-vectorize -fmerge-all-constants -ffast-math

# Set flags for Cubism
CPPFLAGS+= -D_BS_=$(bs) -I$(BUILDDIR)/../Cubism/include/ -DDIMENSION=2

# Set flags for OpenMP and HDF5 libraries
ifeq "$(openmp)" "true"
	CPPFLAGS+= -fopenmp -DUSEOPENMP
	LIBS+= -fopenmp
endif
LIBS+= -lhdf5

#Set VPATH for MAKE to search for files
BUILDDIR = .
VPATH := $(BUILDDIR)/../source/ $(BUILDDIR)/../Cubism/src/
OBJECTS = Simulation.o SimulationData.o Helpers.o ArgumentParser.o advDiff.o AdaptTheMesh.o

# DEFINE COMPILATION TARGETS
all: simulation
.DEFAULT: all;

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f simulation *.o

simulation: main.o $(OBJECTS)
	$(CXX) main.o $(OBJECTS) $(LIBS) -o $@
