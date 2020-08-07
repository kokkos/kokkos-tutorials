cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialTeamScratchMemory)
include(../../common.cmake)

add_executable(TeamScratchMemory team_scratch_memory_begin.cpp)
target_link_libraries(TeamScratchMemory Kokkos::kokkos)


