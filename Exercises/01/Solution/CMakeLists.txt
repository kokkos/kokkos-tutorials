cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial01)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

KokkosTutorials_WarnGPU()

add_executable(01_Solution exercise_1_solution.cpp)
target_link_libraries(01_Solution Kokkos::kokkos)
