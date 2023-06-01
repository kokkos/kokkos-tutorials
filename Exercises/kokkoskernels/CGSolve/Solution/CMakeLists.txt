cmake_minimum_required(VERSION 3.16)
project(KokkosKernelsCGSolve)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(cgsolve cgsolve.cpp)
target_link_libraries(cgsolve Kokkos::kokkoskernels)


