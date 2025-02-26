cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSubview)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(subview_exercise exercise_subview_begin.cpp)
target_link_libraries(subview_exercise Kokkos::kokkos)
