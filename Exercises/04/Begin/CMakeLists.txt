cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial04)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(04_Exercise exercise_4_begin.cpp)
target_link_libraries(04_Exercise Kokkos::kokkos)


