cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialParallelScan)
include(../../common.cmake)

add_executable(ParallelScan_Solution parallel_scan.cpp)
target_link_libraries(ParallelScan_Solution Kokkos::kokkos)

