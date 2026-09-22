cmake_minimum_required(VERSION 3.16)
project(KokkosScalarIntegration LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)

add_executable(scalar_integration scalar_integration.cpp)
target_link_libraries(scalar_integration PRIVATE Kokkos::kokkos)

