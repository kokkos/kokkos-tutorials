cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial03)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(03_Exercise exercise_3_begin.cpp)
target_link_libraries(03_Exercise Kokkos::kokkos)
