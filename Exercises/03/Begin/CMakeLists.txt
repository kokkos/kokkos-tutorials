cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial03)
include(../../common.cmake)

add_executable(03_Exercise exercise_3_begin.cpp)
target_link_libraries(03_Exercise Kokkos::kokkos)


