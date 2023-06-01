cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMPIPackUnpack)
include(../../common.cmake)

add_executable(MPIPackUnpack mpi_pack_unpack_solution.cpp)
target_link_libraries(MPIPackUnpack Kokkos::kokkos)

find_package(MPI REQUIRED)
target_link_libraries(MPIPackUnpack MPI::MPI_CXX)
