cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMD)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(SIMD_Solution simd_solution.cpp)
target_link_libraries(SIMD_Solution Kokkos::kokkos)
