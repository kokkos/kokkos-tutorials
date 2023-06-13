cmake_minimum_required(VERSION 3.16)
project(heat3d)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(heat3d mpi_heat_conduction_solution.cpp)
target_link_libraries(heat3d Kokkos::kokkos)
