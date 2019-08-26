cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial03)
add_executable(03_Exercise exercise_3_solution.cpp)
target_link_libraries(03_Exercise Kokkos::kokkos)


