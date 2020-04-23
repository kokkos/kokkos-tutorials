cmake_minimum_required (VERSION 3.10)
project (KokkosTutorial01)
include(../../common.cmake)

add_executable(01_Exercise exercise_1_solution.cpp)
target_link_libraries(01_Exercise Kokkos::kokkos)


