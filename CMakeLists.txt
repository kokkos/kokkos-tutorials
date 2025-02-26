cmake_minimum_required(VERSION 3.16)
project(KokkosTutorials)

# This option is used to force the use of an internal Kokkos
# It is useful if you have a Kokkos installed but you do not want to use it
# It is especially useful if you have a debian based system with Kokkos installed by apt
# as the CMakeConfig.cmake is partially broken
option(KokkosTutorials_FORCE_INTERNAL_Kokkos "Do not look for any installed Kokkos and use an internal one" OFF)

# This option is used to force the use of an external Kokkos
# It is useful to ensure that the code is compatible with an installed Kokkos
option(KokkosTutorials_FORCE_EXTERNAL_Kokkos "Do not use an internal Kokkos and always look for an installed one" OFF)

if (KokkosTutorials_FORCE_INTERNAL_Kokkos AND KokkosTutorials_FORCE_EXTERNAL_Kokkos)
  message(FATAL_ERROR "You cannot force both internal and external Kokkos:"
          "choose one of KokkosTutorials_FORCE_INTERNAL_Kokkos or KokkosTutorials_FORCE_EXTERNAL_Kokkos")
endif()

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
add_subdirectory(Exercises/fortran-kokkosinterface)  # TODO, Add a check for fortran
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
