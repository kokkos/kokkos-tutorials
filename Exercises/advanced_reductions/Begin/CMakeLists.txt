cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialAdvancedReductions)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(AdvancedReductions advanced_reductions.cpp)
target_link_libraries(AdvancedReductions Kokkos::kokkos)
