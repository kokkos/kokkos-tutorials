cmake_minimum_required(VERSION 3.16)
project(KokkosTutorials)

# These directories follow IntroFull order of introduction

add_subdirectory(Exercises/01)
add_subdirectory(Exercises/02)
add_subdirectory(Exercises/03)
add_subdirectory(Exercises/04)

add_subdirectory(Exercises/dualview)
add_subdirectory(Exercises/mdrange)
add_subdirectory(Exercises/subview)
add_subdirectory(Exercises/scatter_view)
add_subdirectory(Exercises/team_policy)
add_subdirectory(Exercises/team_vector_loop)
add_subdirectory(Exercises/team_scratch_memory)
add_subdirectory(Exercises/simd)
# FIXME update the code
# add_subdirectory(Exercises/simd_warp)
# add_subdirectory(Exercises/tasking)

add_subdirectory(Exercises/advanced_reductions)
add_subdirectory(Exercises/instances)
add_subdirectory(Exercises/multi_gpu_cuda)
add_subdirectory(Exercises/parallel_scan)
add_subdirectory(Exercises/random_number)
add_subdirectory(Exercises/tools_minimd)
add_subdirectory(Exercises/unique_token)
add_subdirectory(Exercises/unordered_map)
add_subdirectory(Exercises/virtualfunction)

# Not done yet
# TODO, require Kokkos Kernels
# add_subdirectory(Exercises/kokkoskernels)
# require remote spaces
# add_subdirectory(Exercises/vectorshift)
