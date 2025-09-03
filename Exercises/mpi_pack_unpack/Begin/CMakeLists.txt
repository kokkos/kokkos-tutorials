cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMPIPackUnpack)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(MPIPackUnpack mpi_pack_unpack_begin.cpp)
target_link_libraries(MPIPackUnpack Kokkos::kokkos)

find_package(MPI REQUIRED)
target_link_libraries(MPIPackUnpack MPI::MPI_CXX)
