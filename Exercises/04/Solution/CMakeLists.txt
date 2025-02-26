cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial04)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(04_Solution exercise_4_solution.cpp)
target_link_libraries(04_Solution Kokkos::kokkos)
