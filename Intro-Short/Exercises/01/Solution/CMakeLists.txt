cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorial01)
add_executable(01_Exercise exercise_1_solution.cpp)
target_link_libraries(01_Exercise Kokkos::kokkos)


