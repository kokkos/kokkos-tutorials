cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialParallelScan)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(ParallelScan parallel_scan.cpp)
target_link_libraries(ParallelScan Kokkos::kokkos)
