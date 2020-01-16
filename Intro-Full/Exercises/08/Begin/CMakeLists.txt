cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial08)
add_executable(08_Exercise exercise_8_begin.cpp)
target_link_libraries(08_Exercise Kokkos::kokkos)


