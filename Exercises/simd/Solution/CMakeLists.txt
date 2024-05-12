cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMD)
include(../../common.cmake)

add_executable(SIMD_Solution simd_solution.cpp)
target_link_libraries(SIMD_Solution Kokkos::kokkos)
