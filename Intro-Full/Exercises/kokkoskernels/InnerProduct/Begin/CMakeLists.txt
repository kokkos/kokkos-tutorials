cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsInnerProduct)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(inner innerproduct.cpp)
target_link_libraries(inner Kokkos::kokkoskernels)


