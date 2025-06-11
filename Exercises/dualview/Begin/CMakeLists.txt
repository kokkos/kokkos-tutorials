cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialDualView)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(dualview dual_view_exercise.cpp)
target_link_libraries(dualview Kokkos::kokkos)
