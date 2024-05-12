cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMDWarp)
include(../../common.cmake)

add_executable(SIMDWarp_Solution simd_warp_solution.cpp)
target_link_libraries(SIMDWarp_Solution Kokkos::kokkos)
