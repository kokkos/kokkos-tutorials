cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial01)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

KokkosTutorials_WarnGPU()

add_executable(01_Exercise exercise_1_begin.cpp)
target_link_libraries(01_Exercise Kokkos::kokkos)
