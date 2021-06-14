cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsInnerProduct)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(innerproduct.exe innerproduct.cpp)
target_link_libraries(innerproduct.exe Kokkos::kokkoskernels)

