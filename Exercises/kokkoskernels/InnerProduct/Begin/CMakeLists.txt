cmake_minimum_required(VERSION 3.16)
project(KokkosKernelsInnerProduct)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(innerproduct.exe innerproduct.cpp)
target_link_libraries(innerproduct.exe Kokkos::kokkoskernels)

