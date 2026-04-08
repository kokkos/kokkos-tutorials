cmake_minimum_required(VERSION 3.22)
project(KokkosTutorialExecGraph)
include(../../common.cmake)

add_executable(graph graph_begin.cpp)
target_link_libraries(graph Kokkos::kokkos)
