cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMDWarp)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(SIMDWarp_Solution simd_warp_solution.cpp)
target_link_libraries(SIMDWarp_Solution Kokkos::kokkos)
