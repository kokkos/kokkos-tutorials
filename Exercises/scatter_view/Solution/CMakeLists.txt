cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialScatterView)
include(../../common.cmake)

add_executable(scatterview_Solution scatter_view.cpp)
target_link_libraries(scatterview_Solution Kokkos::kokkos)
