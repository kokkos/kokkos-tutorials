cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial04)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(04_Solution exercise_4_solution.cpp)
target_link_libraries(04_Solution Kokkos::kokkos)
