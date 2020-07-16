cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial04)
include(../../common.cmake)

add_executable(04_Exercise exercise_4_begin.cpp)
target_link_libraries(04_Exercise Kokkos::kokkos)


