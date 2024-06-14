cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSubview)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(subview_Solution exercise_subview_solution.cpp)
target_link_libraries(subview_Solution Kokkos::kokkos)
