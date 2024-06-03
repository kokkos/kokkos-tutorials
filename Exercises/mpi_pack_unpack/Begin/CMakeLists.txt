cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMPIPackUnpack)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(MPIPackUnpack mpi_pack_unpack_begin.cpp)
target_link_libraries(MPIPackUnpack Kokkos::kokkos)

find_package(MPI REQUIRED)
target_link_libraries(MPIPackUnpack MPI::MPI_CXX)
