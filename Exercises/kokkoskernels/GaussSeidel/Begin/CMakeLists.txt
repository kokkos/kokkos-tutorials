cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsGaussSeidel)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(gauss_seidel gauss_seidel.cpp)
target_link_libraries(gauss_seidel Kokkos::kokkoskernels)


