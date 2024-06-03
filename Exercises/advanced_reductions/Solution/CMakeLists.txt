cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialAdvancedReductions)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(AdvancedReductions_Solution advanced_reductions.cpp)
target_link_libraries(AdvancedReductions_Solution Kokkos::kokkos)
