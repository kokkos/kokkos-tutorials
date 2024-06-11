cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamScratchMemory)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(TeamScratchMemory team_scratch_memory_begin.cpp)
target_link_libraries(TeamScratchMemory Kokkos::kokkos)
