cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamVectorLoop)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(TeamVectorLoop team_vector_loop_begin.cpp)
target_link_libraries(TeamVectorLoop Kokkos::kokkos)
