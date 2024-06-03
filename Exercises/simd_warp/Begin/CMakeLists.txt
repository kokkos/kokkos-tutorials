cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSIMDWarp)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(SIMDWarp simd_warp_begin.cpp)
target_link_libraries(SIMDWarp Kokkos::kokkos)
