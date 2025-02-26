cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialScatterView)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(scatterview scatter_view.cpp)
target_link_libraries(scatterview Kokkos::kokkos)
