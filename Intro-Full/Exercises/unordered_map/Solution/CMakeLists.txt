cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialUnorderedMap)
include(../../common.cmake)

add_executable(unordered_map unordered_map.cpp)
target_link_libraries(unordered_map Kokkos::kokkos)


