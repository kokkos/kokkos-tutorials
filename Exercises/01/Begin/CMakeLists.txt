cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial01)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(01_Exercise exercise_1_begin.cpp)
target_link_libraries(01_Exercise Kokkos::kokkos)
