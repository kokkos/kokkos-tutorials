cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorialScatterView)
add_executable(scatterview fe_scatter.cpp)
target_link_libraries(scatterview Kokkos::kokkos)


