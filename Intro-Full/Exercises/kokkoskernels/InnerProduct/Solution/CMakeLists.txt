cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosKernelsInnerProduct)
add_executable(inner innerproduct.cpp)

target_link_libraries(inner Kokkos::kokkoskernels)


