cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial02)
include(../../common.cmake)

add_executable(02_Exercise exercise_2_begin.cpp)
target_link_libraries(02_Exercise Kokkos::kokkos)


