cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsSpGEMM)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(spgemm spgemm.cpp)
target_link_libraries(spgemm Kokkos::kokkoskernels)


