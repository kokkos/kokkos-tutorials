cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial04)
add_executable(04_Exercise exercise_4_solution.cpp)
target_link_libraries(04_Exercise Kokkos::kokkos)


