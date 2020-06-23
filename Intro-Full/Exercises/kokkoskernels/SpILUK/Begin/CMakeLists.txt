cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsSpILUK)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(sparse_iluk sparse_iluk.cpp)
target_link_libraries(sparse_iluk Kokkos::kokkoskernels)


