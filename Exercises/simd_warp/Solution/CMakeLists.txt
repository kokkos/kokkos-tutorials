cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMDWarp)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

find_package(Kokkos REQUIRED)

add_executable(SIMDWarp_Solution simd_warp_solution.cpp)
target_link_libraries(SIMDWarp_Solution Kokkos::kokkos)
