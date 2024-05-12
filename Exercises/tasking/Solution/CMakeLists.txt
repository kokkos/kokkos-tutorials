cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTasking)
include(../../common.cmake)

add_executable(Tasking_Solution tasking_solution.cpp)
target_link_libraries(Tasking_Solution Kokkos::kokkos)
