cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTasking)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(Tasking_Solution tasking_solution.cpp)
target_link_libraries(Tasking_Solution Kokkos::kokkos)
