cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialUnorderedMap)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(unordered_map_solution unordered_map.cpp)
target_link_libraries(unordered_map_solution Kokkos::kokkos)
