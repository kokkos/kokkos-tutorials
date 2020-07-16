KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = "Cuda,OpenMP"
EXE_NAME = "dualview"


SRC = $(wildcard *.cpp)
HDR = $(wildcard *.h)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "BDW,Pascal60"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
endif

#CXXFLAGS = -O3 -DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK 
CXXFLAGS = -O3 -UKOKKOS_ENABLE_DEPRECATED_CODE

override CXXFLAGS += -I./
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

vpath %.cpp $(sort $(dir $(SRC)))

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

$(warning $(OBJ) $(EXE) $(sort $(dir $(SRC))))

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS) 
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.a *.h *.tmp *.x

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HDR)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

test: $(EXE)
	./$(EXE)
