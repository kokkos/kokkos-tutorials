cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialSubview)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(subview_exercise exercise_subview_begin.cpp)
target_link_libraries(subview_exercise Kokkos::kokkos)
