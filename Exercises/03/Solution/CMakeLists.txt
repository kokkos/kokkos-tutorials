cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial03)
include(../../common.cmake)

add_executable(03_Solution exercise_3_solution.cpp)
target_link_libraries(03_Solution Kokkos::kokkos)


