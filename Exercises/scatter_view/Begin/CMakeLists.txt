cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialScatterView)
include(../../common.cmake)

add_executable(scatterview fe_scatter.cpp)
target_link_libraries(scatterview Kokkos::kokkos)


