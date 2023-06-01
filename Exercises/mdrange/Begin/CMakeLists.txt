cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMdRange)
include(../../common.cmake)

add_executable(mdrange_exercise exercise_mdrange_begin.cpp)
target_link_libraries(mdrange_exercise Kokkos::kokkos)


