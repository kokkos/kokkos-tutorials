KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOSKERNELS_PATH = ${HOME}/Kokkos/kokkos-kernels
KOKKOS_DEVICES = "Cuda,OpenMP"
EXE_NAME = "inner"

# Turn of ETI
KOKKOSKERNELS_SCALARS =

SRC = $(wildcard *.cpp)
HDR = $(wildcard *.hpp)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "BDW,Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
KOKKOS_ARCH = "POWER8"
endif

#CXXFLAGS = -O3 -DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
CXXFLAGS = -O3
override CXXFLAGS += -I./
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

vpath %.cpp $(sort $(dir $(SRC)))

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos
include $(KOKKOSKERNELS_PATH)/Makefile.kokkos-kernels

$(warning $(OBJ) $(EXE) $(sort $(dir $(SRC))))

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(KOKKOSKERNELS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(KOKKOSKERNELS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.a *.h *.tmp *.x

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HDR)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOSKERNELS_CPPFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

test: $(EXE)
	./$(EXE)
