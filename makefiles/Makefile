bs ?= 8
openmp ?= false

# SET FLAGS FOR COMPILER
ifneq ($(MPICXX),)
  CXX=$(MPICXX)
else
  ERROR += "MPICXX not set"
endif

#ifneq ($(ERROR),)
#	$(error $(ERROR))
#endif

# LOAD FLAGS -- GCC NEEDED
CPPFLAGS+= -std=c++17 -Wall -g
CPPFLAGS+= -Wextra -Wfloat-equal -Wcast-align -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Wwrite-strings -Wno-unused-parameter #-Wno-cpp
CPPFLAGS+= -Wno-float-equal
CPPFLAGS+= -Wno-redundant-decls
CPPFLAGS+= -Wno-unknown-pragmas

# SET FLAGS FOR CUBISM
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -I$(BUILDDIR)/../Cubism/include/ -DDIMENSION=2 -DCUBISM_USE_MAP

#Optimizations
CPPFLAGS+= -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native -falign-functions -ftree-vectorize -fmerge-all-constants -ffast-math

ifeq "$(openmp)" "true"
	CPPFLAGS+= -fopenmp
endif

# FLAGS FOR EXTERNAL LIBRARIES
LIBS+= -fopenmp -lhdf5

# ADD LIBRARY PATHS IF GIVEN
ifneq ($(HDF5_ROOT),)
	LIBS     += -L$(HDF5_ROOT)/lib
	CPPFLAGS += -I$(HDF5_ROOT)/include
endif


## SET VPATH FOR MAKE TO SEARCH FOR FILES
BUILDDIR = .
DIRS = $(sort $(dir $(wildcard ../source/*/)))
VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/

OBJECTS = Simulation.o SimulationData.o Helpers.o ArgumentParser.o advDiff.o AdaptTheMesh.o

# DEFINE COMPILATION TARGETS
all: simulation
.DEFAULT: all;

simulation: main.o $(OBJECTS)
	$(CXX) main.o $(OBJECTS) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CPPFLAGS) -c -MD $<

clean:
	rm -f simulation *.o
