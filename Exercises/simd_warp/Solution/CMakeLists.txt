cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialSIMDWarp)
include(../../common.cmake)

add_executable(SIMDWarp simd_warp_solution.cpp)
target_link_libraries(SIMDWarp Kokkos::kokkos)
