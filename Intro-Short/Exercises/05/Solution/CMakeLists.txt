cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial05)
include(../../common.cmake)

add_executable(05_Exercise exercise_5_solution.cpp)
target_link_libraries(05_Exercise Kokkos::kokkos)


