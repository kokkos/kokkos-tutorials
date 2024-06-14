cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamVectorLoop)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(TeamVectorLoop_Solution team_vector_loop_solution.cpp)
target_link_libraries(TeamVectorLoop_Solution Kokkos::kokkos)
