cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialExercices)

add_subdirectory(01)
add_subdirectory(02)
add_subdirectory(03)
add_subdirectory(04)

add_subdirectory(dualview)
add_subdirectory(mdrange)
add_subdirectory(subview)
add_subdirectory(scatter_view)
add_subdirectory(team_policy)
add_subdirectory(team_vector_loop)
add_subdirectory(team_scratch_memory)
add_subdirectory(tasking)
add_subdirectory(simd)
# FIXME update the code
# add_subdirectory(simd_warp)

add_subdirectory(advanced_reductions)
add_subdirectory(parallel_scan)
