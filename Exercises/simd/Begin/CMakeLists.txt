cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMD)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(SIMD simd_begin.cpp)
target_link_libraries(SIMD Kokkos::kokkos)
