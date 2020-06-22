cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial09)
include(../../common.cmake)

add_executable(09_Exercise exercise_9_solution.cpp)
target_link_libraries(09_Exercise Kokkos::kokkos)


