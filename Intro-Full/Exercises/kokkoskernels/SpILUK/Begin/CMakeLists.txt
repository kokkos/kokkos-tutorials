cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsSpILUK)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

if(NOT DEFINED KokkosKernels_SOURCE_DIR)
 message( FATAL_ERROR "Please specify KokkosKernels_SOURCE_DIR to include necessary test headers." )
endif()

include_directories(${KokkosKernels_SOURCE_DIR}/test_common)

add_executable(sparse_iluk sparse_iluk.cpp)
target_link_libraries(sparse_iluk Kokkos::kokkoskernels)


