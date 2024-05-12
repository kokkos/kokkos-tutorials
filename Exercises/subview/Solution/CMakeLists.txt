cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSubview)
include(../../common.cmake)

add_executable(subview_Solution exercise_subview_solution.cpp)
target_link_libraries(subview_Solution Kokkos::kokkos)
