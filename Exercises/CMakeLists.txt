cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialExercices)

# These directories follow IntroFull order of introduction

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

# These directories are "independent" and  listed in alphabetical order

add_subdirectory(advanced_reductions)
add_subdirectory(fortran-kokkosinterface)  # TODO, Add a check for fortran
add_subdirectory(instances)
add_subdirectory(multi_gpu_cuda)
add_subdirectory(parallel_scan)
add_subdirectory(tools_minimd)
add_subdirectory(random_number)
add_subdirectory(unique_token)
add_subdirectory(unordered_map)
add_subdirectory(virtualfunction)

# Not done yet
# TODO, require Kokkos Kernels
# add_subdirectory(kokkoskernels)
# require remote spaces
# add_subdirectory(vectorshift)