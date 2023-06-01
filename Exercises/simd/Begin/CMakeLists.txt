cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMD)
include(../../common.cmake)

add_executable(SIMD simd_begin.cpp)
target_link_libraries(SIMD Kokkos::kokkos)
