cmake_minimum_required(VERSION 3.16)
project(mpi_exch)

# Add a custom module path for find_package(Kokkos)
include(../cmake/SetUpKokkos.cmake)

find_package(MPI REQUIRED)

add_executable(mpi_exch mpi_exch.cpp)
target_link_libraries(mpi_exch Kokkos::kokkos)
target_link_libraries(mpi_exch MPI::MPI_CXX)
