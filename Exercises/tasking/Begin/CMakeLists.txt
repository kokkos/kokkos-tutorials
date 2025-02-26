cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTasking)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

find_package(Kokkos REQUIRED)

add_executable(Tasking tasking_begin.cpp)
target_link_libraries(Tasking Kokkos::kokkos)
