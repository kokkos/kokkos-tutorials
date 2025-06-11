cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial04)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(04_Exercise exercise_4_begin.cpp)
target_link_libraries(04_Exercise Kokkos::kokkos)
