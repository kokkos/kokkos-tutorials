cmake_minimum_required(VERSION 3.16)
project(KokkosTutorial02)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL OR Kokkos_ENABLE_OPENMPTARGET OR Kokkos_ENABLE_HPX)
    message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}"
            "A Kokkos accelerator backend is enabled, it might cause issue with the current program"
            "Please recompile with only a host backend enabled (e.g. -DKokkos_ENABLE_OPENMP=ON)")
endif ()

add_executable(02_Solution exercise_2_solution.cpp)
target_link_libraries(02_Solution Kokkos::kokkos)
