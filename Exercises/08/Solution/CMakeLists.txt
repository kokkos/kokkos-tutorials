cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial08)
include(../../common.cmake)

add_executable(08_Exercise exercise_8_solution.cpp)
target_link_libraries(08_Exercise Kokkos::kokkos)


