cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial07)
add_executable(07_Exercise exercise_7_solution.cpp)
target_link_libraries(07_Exercise Kokkos::kokkos)


