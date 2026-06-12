cmake_minimum_required(VERSION 3.16)
project(heat3d)

set(KOKKOS_TAG CACHE STRING "Kokkos version tag")

include(FetchContent)
if(NOT KOKKOS_TAG STREQUAL "")
  message(STATUS "KOKKOS_TAG=${KOKKOS_TAG}")
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        ${KOKKOS_TAG}
    FIND_PACKAGE_ARGS NAMES Kokkos::kokkos
  )
else()
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    FIND_PACKAGE_ARGS NAMES Kokkos::kokkos
  )
endif()
FetchContent_MakeAvailable(Kokkos)

find_package(MPI REQUIRED)

add_executable(heat3d mpi_heat_conduction_solution.cpp)
target_link_libraries(heat3d Kokkos::kokkos)
target_link_libraries(heat3d MPI::MPI_CXX)
