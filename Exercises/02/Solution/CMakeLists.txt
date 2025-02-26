cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial02)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

KokkosTutorials_WarnGPU()

add_executable(02_Solution exercise_2_solution.cpp)
target_link_libraries(02_Solution Kokkos::kokkos)
