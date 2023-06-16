cmake_minimum_required(VERSION 3.16)
project(mpi_exch)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(mpi_exch mpi_exch.cpp)
target_link_libraries(mpi_exch Kokkos::kokkos)
