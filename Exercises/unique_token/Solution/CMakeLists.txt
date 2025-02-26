cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialUniqueToken)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(uniquetoken_solution unique_token.cpp)
target_link_libraries(uniquetoken_solution Kokkos::kokkos)
