cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial03)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(03_Solution exercise_3_solution.cpp)
target_link_libraries(03_Solution Kokkos::kokkos)
