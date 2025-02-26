cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialUnorderedMap)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

find_package(Kokkos REQUIRED)

add_executable(unordered_map_exercise unordered_map.cpp)
target_link_libraries(unordered_map_exercise Kokkos::kokkos)
