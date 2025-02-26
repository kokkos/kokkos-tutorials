cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSubview)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(subview_Solution exercise_subview_solution.cpp)
target_link_libraries(subview_Solution Kokkos::kokkos)
