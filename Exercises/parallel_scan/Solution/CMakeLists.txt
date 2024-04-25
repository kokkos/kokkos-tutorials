cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial01)
include(../../common.cmake)

add_executable(ParallelScan parallel_scan.cpp)
target_link_libraries(ParallelScan Kokkos::kokkos)

