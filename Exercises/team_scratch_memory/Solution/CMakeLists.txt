cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamScratchMemory)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(TeamScratchMemory_Solution team_scratch_memory_solution.cpp)
target_link_libraries(TeamScratchMemory_Solution Kokkos::kokkos)
