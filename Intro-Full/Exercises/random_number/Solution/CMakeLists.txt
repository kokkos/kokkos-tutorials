cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorialRNG)
add_executable(MC_DartSampler MC_DartSampler.cpp)
target_link_libraries(MC_DartSampler Kokkos::kokkos)


