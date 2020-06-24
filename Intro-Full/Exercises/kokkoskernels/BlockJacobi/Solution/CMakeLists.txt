cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsBlockjacobi)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(blockjacobi blockjacobi.cpp)
target_link_libraries(blockjacobi Kokkos::kokkoskernels)


