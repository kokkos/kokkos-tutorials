cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTasking)
include(../../common.cmake)

add_executable(Median median_finding.cpp)
target_link_libraries(Median Kokkos::kokkos)
