cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTasking)
include(../../common.cmake)

add_executable(Tasking tasking_solution.cpp)
target_link_libraries(Tasking Kokkos::kokkos)
