cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialFortran)
include(../../common.cmake)

set(SPACK_FC $ENV{SPACK_FC})
if(SPACK_FC)
  message("Found spack fortran compiler ${SPACK_FC}")
  set(CMAKE_Fortran_COMPILER ${SPACK_FC} CACHE FILEPATH "the C++ compiler" FORCE)
  set(ENV{FC} ${SPACK_FC})
endif()

set(SPACK_F77 $ENV{SPACK_F77})
if(SPACK_F77)
  message("Found spack fortran compiler ${SPACK_F77}")
  set(CMAKE_Fortran_COMPILER ${SPACK_F77} CACHE FILEPATH "the C++ compiler" FORCE)
  set(ENV{F77} ${SPACK_F77})
endif()

execute_process (
    COMMAND bash -c "printenv"
    OUTPUT_VARIABLE outVar
)
message("${outVar}")

enable_language(Fortran)

add_executable(ftest.x abi.f90 f_interface.f90 main.f90 c_interface.cpp)
target_link_libraries(ftest.x Kokkos::kokkos)


