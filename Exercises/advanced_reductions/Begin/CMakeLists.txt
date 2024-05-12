cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialAdvancedReductions)
include(../../common.cmake)

add_executable(AdvancedReductions advanced_reductions.cpp)
target_link_libraries(AdvancedReductions Kokkos::kokkos)

