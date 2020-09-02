#Set the path to MPI
MPI_PATH = 
#Use MPI
HAVE_MPI = no

#Set the path to Kokkos
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
#Set the Devices to compile for
KOKKOS_DEVICES=OpenMP
#Set the Architecture to compiler for
KOKKOS_ARCH=SNB,Volta70
#Set third party library usage
KOKKOS_USE_TPL=
CXXFLAGS = -O3 -g 
LINKFLAGS = 
LIB =

ifeq ($(HAVE_MPI), yes)
CXX = mpicxx
else
CXX = g++
CXXFLAGS += -I./MPI-Stubs
LIB += libmpi_stubs.a
endif

CXXFLAGS += -DKOKKOSP_ENABLE_PROFILING

LINK = ${CXX}
EXE = miniMD

#CXXFLAGS += -DTEST_LAMBDA_BYCOPY
#CXXFLAGS += -DTEST_LAMBDA_BYPTR

#Check for single precision
ifeq ($(SP), yes)
CXXFLAGS += -DPRECISION=1 
LINKFLAGS += -DPRECISION=1 
else
CXXFLAGS += -DPRECISION=2
LINKFLAGS += -DPRECISION=2
endif

#Check if debug on
ifeq ($(DEBUG), yes)
CXXFLAGS += -g -G 
LINKFLAGS += -g 
endif

# Check for KNC compile
ifeq ($(KNC), yes)
override KOKKOS_ARCH=KNC
endif

# Check for KNC compile
ifeq ($(AVX), yes)
override KOKKOS_ARCH=SNB
endif

#Use LIBRT for precision timers
ifeq ($(LIBRT),yes)
CXXFLAGS += -DPREC_TIMER
KOKKOS_USE_TPL += ,librt
endif

#Check for PAD
ifeq ($(PAD), 3)
CXXFLAGS += -DPAD3 
endif
ifeq ($(PAD), 4)
CXXFLAGS += -DPAD4 
endif

#Check for use of explicit SIMD pragma
ifeq ($(SIMD), yes)
CXXFLAGS += -DUSE_SIMD
endif

#Check for request of ANSI_ALIAS
ifeq ($(ANSI_ALIAS), yes)
CXXFLAGS += -ansi-alias 
endif


ifeq ($(RED_PREC), yes)
ifeq ($(KOKKOS_ARCH), KNC)
CXXFLAGS += -fimf-precision=low -fimf-domain-exclusion=15 
else
CXXFLAGS += -mGLOB_default_function_attrs="use_approx_f64_divide=true"
endif
endif

ifeq ($(GSUNROLL), yes)
ifeq ($(KOKKOS_ARCH), KNC)
CXXFLAGS += -mGLOB_default_function_attrs="gather_scatter_loop_unroll=7; use_gather_scatter_hint=on"
endif
endif

default: all

include $(KOKKOS_PATH)/Makefile.kokkos

SRC = $(wildcard *.cpp) 
HEADERS = $(wildcard *.h)
DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)

mpi.o:
	$(CC) -O -fPIC -c MPI-Stubs/mpi.c

libmpi_stubs.a: mpi.o
	ar rs libmpi_stubs.a mpi.o 

all: $(OBJ) $(KOKKOS_LINK_DEPENDS) libmpi_stubs.a
	$(LINK) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm *.o *.d libkokkos.a *.cuda *.host mpi.o libmpi_stubs.a

EXTRA_FLAGS = $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)
# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c $<


# Test
scope=0
input=lj
halfneigh=0
path=""

test:
	bash run_tests ${scope} ${input} ${halfneigh} ${path}
