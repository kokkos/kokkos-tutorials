cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial02)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

if (Kokkos_DEVICES)
    message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}"
            "Kokkos device is enabled, it might cause issue with the current program")
endif ()

add_executable(02_Exercise exercise_2_begin.cpp)
target_link_libraries(02_Exercise Kokkos::kokkos)


