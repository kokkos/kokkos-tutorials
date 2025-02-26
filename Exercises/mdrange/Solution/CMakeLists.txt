cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMdRange)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(mdrange_Solution exercise_mdrange_solution.cpp)
target_link_libraries(mdrange_Solution Kokkos::kokkos)
