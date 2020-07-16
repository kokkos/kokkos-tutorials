cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial06)
include(../../common.cmake)

add_executable(06_Exercise exercise_6_begin.cpp)
target_link_libraries(06_Exercise Kokkos::kokkos)


