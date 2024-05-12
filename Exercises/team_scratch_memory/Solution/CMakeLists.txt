cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamScratchMemory)
include(../../common.cmake)

add_executable(TeamScratchMemory_Solution team_scratch_memory_solution.cpp)
target_link_libraries(TeamScratchMemory_Solution Kokkos::kokkos)
