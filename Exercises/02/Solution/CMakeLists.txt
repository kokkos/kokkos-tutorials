cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial02)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

if (Kokkos_DEVICES)
    message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}"
            "Kokkos device is enabled, it might cause issue with the current program")
endif ()

add_executable(02_Solution exercise_2_solution.cpp)
target_link_libraries(02_Solution Kokkos::kokkos)
