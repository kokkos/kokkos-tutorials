cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosKernelsCGSolve)
add_executable(cgsolve cgsolve.cpp)

target_link_libraries(cgsolve Kokkos::kokkoskernels)


