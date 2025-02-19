#!/bin/bash

set -eou pipefail

tutorials_src="$1"
backend="$2"

# These are exercises with executables that can be run in the Solution subdirectory
# TODO: advanced_reductions seems broken
# TODO: hpcbind does not use cmake
# TODO: instances does not use cmake
# TODO: vectorshift needs Kokkos Remote Spaces
# TODO: kokkoskernels/CGSolve_SpILUKprecond needs to know where Kokkos Kernels source directory is
# TODO: kokkoskernels/SpILUK needs to know where Kokkos Kernels source directory is
# TODO: kokkoskernels/TeamGemm seems broken
# TODO: mpi_heat_conduction/no-mpi does not use cmake
# TODO: mpi_pack_unpack need to be run with MPI
SOLUTION_EXERCISES=(
01
02
03
dualview
kokkoskernels/BlockJacobi
kokkoskernels/GaussSeidel
kokkoskernels/GraphColoring
kokkoskernels/InnerProduct
mdrange
parallel_scan
random_number
scatter_view
simd
subview
team_policy
team_scratch_memory
team_vector_loop
unordered_map
)

if [ "$backend" == CUDA ]; then
  SOLUTION_EXERCISES+=(04)
  SOLUTION_EXERCISES+=(multi_gpu_cuda)
fi

if [ ! "$backend" == CUDA ]; then
  SOLUTION_EXERCISES+=(tasking)
  SOLUTION_EXERCISES+=(virtualfunction)
fi

if [ "$backend" == OPENMP ]; then
  SOLUTION_EXERCISES+=(unique_token)
  export OMP_PROC_BIND=spread
  export OMP_PLACES=threads
fi

if [[ ! "$OSTYPE" == "darwin"* ]]; then
  SOLUTION_EXERCISES+=(fortran-kokkosinterface)
fi

for e in "${SOLUTION_EXERCISES[@]}"; do
solution_dir="build/Exercises/$e/Solution"
  if [ -d "$solution_dir" ]; then
    # Executable doesen't follow a naming convention
    for executable in "$solution_dir"/*; do
      if [ -x "$executable" ] && [ ! -d "$executable" ]; then
        echo "Running $executable"
        "$executable"
      fi
    done
  fi
done
