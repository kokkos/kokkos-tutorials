cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialSubview)
include(../../common.cmake)

add_executable(subview_exercise exercise_subview_begin.cpp)
target_link_libraries(subview_exercise Kokkos::kokkos)


