cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial03)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(03_Exercise exercise_3_begin.cpp)
target_link_libraries(03_Exercise Kokkos::kokkos)


