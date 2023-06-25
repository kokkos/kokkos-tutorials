cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTools)
include(../common.cmake)

option(ENABLE_MPI OFF "Whether to enable the Message Passing Interface (MPI)")

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
else()
  add_library(mpi_stubs MPI-Stubs/mpi.c)
  target_include_directories(mpi_stubs PUBLIC MPI-Stubs)
endif()

add_executable(miniMD
  atom.cpp
  comm.cpp
  force_eam.cpp
  force_lj.cpp
  input.cpp
  integrate.cpp
  ljs.cpp
  neighbor.cpp
  output.cpp
  setup.cpp
  thermo.cpp
  timer.cpp
)

target_link_libraries(miniMD Kokkos::kokkos)

if(ENABLE_MPI)
  target_link_libraries(miniMD MPI::MPI_CXX)
else()
  target_link_libraries(miniMD mpi_stubs)
endif()

