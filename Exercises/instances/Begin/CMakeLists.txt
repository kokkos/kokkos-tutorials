cmake_minimum_required(VERSION 3.22)
project(KokkosTutorialExecInstances)
include(../../common.cmake)

add_executable(instances instances_begin.cpp)
target_link_libraries(instances Kokkos::kokkos)
