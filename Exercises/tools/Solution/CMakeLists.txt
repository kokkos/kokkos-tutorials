cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTools)
include(../../common.cmake)

add_executable(tools_exercise exercise_tools_solution.cpp)
target_link_libraries(tools_exercise Kokkos::kokkos)


