cmake_minimum_required(VERSION 3.22)
project(KokkosTutorialConcepts)
include(../../common.cmake)

add_executable(concepts concepts.cpp)
target_link_libraries(concepts Kokkos::kokkos)

