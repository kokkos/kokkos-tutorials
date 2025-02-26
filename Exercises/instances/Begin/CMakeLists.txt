cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialInstances)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(Instances_exercise instances_begin.cpp)
target_link_libraries(Instances_exercise Kokkos::kokkos)
