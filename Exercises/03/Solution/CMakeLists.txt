cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial03)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(03_Solution exercise_3_solution.cpp)
target_link_libraries(03_Solution Kokkos::kokkos)
