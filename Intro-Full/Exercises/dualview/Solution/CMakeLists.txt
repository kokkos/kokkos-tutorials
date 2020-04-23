cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialDualView)
include(../../common.cmake)

add_executable(dualview dual_view_exercise.cpp)
target_link_libraries(dualview Kokkos::kokkos)


