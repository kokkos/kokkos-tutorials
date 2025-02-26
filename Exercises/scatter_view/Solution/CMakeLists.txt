cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialScatterView)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(scatterview_Solution scatter_view.cpp)
target_link_libraries(scatterview_Solution Kokkos::kokkos)
