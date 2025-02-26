cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialParallelScan)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(ParallelScan parallel_scan.cpp)
target_link_libraries(ParallelScan Kokkos::kokkos)
