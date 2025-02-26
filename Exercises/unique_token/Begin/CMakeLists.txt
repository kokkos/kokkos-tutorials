cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialUniqueToken)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

if (NOT Kokkos_ENABLE_OPENMP)
  message(WARNING "This exercise requires OpenMP, enable with -DKokkos_ENABLE_OPENMP=ON")
  return()
endif()

add_executable(uniquetoken_exercise unique_token.cpp)
target_link_libraries(uniquetoken_exercise Kokkos::kokkos)
