KOKKOS_DEVICES=OpenMP
KOKKOS_CUDA_OPTIONS=enable_lambda
KOKKOS_ARCH = "SNB,Volta70"


MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

ifndef KOKKOS_PATH
  KOKKOS_PATH = ${HOME}/Kokkos/kokkos
endif

SRC = $(wildcard $(MAKEFILE_PATH)*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

LINKFLAGS =
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = scatter_view.cuda
override LINKFLAGS += --remove-duplicate-link-files
else
CXX = g++
EXE = scatter_view.host
endif

CXXFLAGS ?= -O3 -g
override CXXFLAGS += -I$(MAKEFILE_PATH)

DEPFLAGS = -M
LINK = ${CXX}

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(info $$LINK is [${LINK}])
$(info $$KOKKOS_LDFLAGS is [${KOKKOS_LDFLAGS}])
$(info $$LINKFLAGS is [${LINKFLAGS}])
$(info $$EXTRA_PATH is [${EXTRA_PATH}])
$(info $$OBJ is [${OBJ}])
$(info $$KOKKOS_LIBS is [${KOKKOS_LIBS}])
$(info $$LIB is [${LIB}])
$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
