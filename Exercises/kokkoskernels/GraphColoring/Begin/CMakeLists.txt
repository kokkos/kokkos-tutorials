cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsGraphColoring)

find_package(KokkosKernels REQUIRED)

include(../../common.cmake)

add_executable(graph_coloring graph_coloring.cpp)
target_link_libraries(graph_coloring Kokkos::kokkoskernels)


