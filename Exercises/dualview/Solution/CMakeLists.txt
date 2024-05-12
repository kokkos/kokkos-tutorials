cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialDualView)
include(../../common.cmake)

add_executable(dualview_Solution dual_view_exercise.cpp)
target_link_libraries(dualview_Solution Kokkos::kokkos)
