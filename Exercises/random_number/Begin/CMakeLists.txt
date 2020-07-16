cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialRNG)
include(../../common.cmake)

add_executable(MC_DartSampler MC_DartSampler.cpp)
target_link_libraries(MC_DartSampler Kokkos::kokkos)


