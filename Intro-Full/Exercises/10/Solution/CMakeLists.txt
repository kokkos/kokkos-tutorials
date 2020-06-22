cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial10)
include(../../common.cmake)

add_executable(10_Exercise exercise_10_solution.cpp)
target_link_libraries(10_Exercise Kokkos::kokkos)


