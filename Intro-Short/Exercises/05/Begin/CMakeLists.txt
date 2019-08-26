cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial05)
add_executable(05_Exercise exercise_5_begin.cpp)
target_link_libraries(05_Exercise Kokkos::kokkos)


