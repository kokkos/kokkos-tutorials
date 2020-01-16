cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial06)
add_executable(06_Exercise exercise_6_solution.cpp)
target_link_libraries(06_Exercise Kokkos::kokkos)


