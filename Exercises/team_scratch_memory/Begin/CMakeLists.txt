cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamScratchMemory)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(TeamScratchMemory team_scratch_memory_begin.cpp)
target_link_libraries(TeamScratchMemory Kokkos::kokkos)
