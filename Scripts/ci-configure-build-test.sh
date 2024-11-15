#! /bin/bash

set -eou pipefail

kokkos_root="$1"
tutorials_src="$2"
cpp_compiler="$3"
build_type="$4"
backend="$5"

# These are exercises with CMakeLists.txt in Begin and Solution subdirectories
# TODO: advanced_reductions seems broken
# TODO: hpcbind does not use cmake
# TODO: instances does not use cmake
# TODO: kokkoskernels/BlockJacobi requires kokkos-kernels
# TODO: kokkoskernels/CGSolve requires kokkos-kernels
# TODO: kokkoskernels/CGSolve_SpILUKprecond requires kokkos-kernels
# TODO: kokkoskernels/GaussSeidel requires kokkos-kernels
# TODO: kokkoskernels/GraphColoring requires kokkos-kernels
# TODO: kokkoskernels/InnerProduct requires kokkos-kernels
# TODO: kokkoskernels/SpGEMM requires kokkos-kernels
# TODO: kokkoskernels/SpILUK requires kokkos-kernels
# TODO: kokkoskernels/TeamGemm requires kokkos-kernels
# TODO: mpi_exch needs MPI
# TODO: mpi_heat_conduction needs MPI
# TODO: mpi_pack_unpack needs MPI
# TODO: parallel_scan seems broken
# TODO: simd_warp seems broken
# TODO: subview seems broken
# TODO: vectorshift needs Kokkos Remote Spaces
BEGIN_SOLUTION_EXERCISES=(
01
02
03
dualview
mdrange
random_number
scatter_view
simd
tasking
team_policy
team_scratch_memory
team_vector_loop
unordered_map
virtualfunction
)

if [ "$backend" == CUDA ]; then
  BEGIN_SOLUTION_EXERCISES+=(04)
  BEGIN_SOLUTION_EXERCISES+=(multi_gpu_cuda)
fi
if [ "$backend" == OPENMP ]; then
  BEGIN_SOLUTIONS_EXERCISES+=(unique_token)
fi

# no fortran on macOs
if [[ ! "$OSTYPE" == "darwin"* ]]; then
    BEGIN_SOLUTIONS_EXERCISES+=(fortran-kokkosinterface)
fi

# These are exercises with CMakeLists.txt in the root directory
EXERCISES=(
tools_minimd
)

# Add Begin and Solution subdirectory from BEGIN_SOLUTION_EXCERCISES to EXERCISES
for e in "${BEGIN_SOLUTION_EXERCISES[@]}"; do
  EXERCISES+=("$e"/Begin)
  EXERCISES+=("$e"/Solution)
done

export Kokkos_ROOT="$kokkos_root"
mkdir -p build
for e in "${EXERCISES[@]}"; do
  source_dir="$tutorials_src"/Exercises/"$e"
  build_dir=build/Exercises/"$e"
  echo building "$source_dir"
  cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_CXX_COMPILER="$cpp_compiler" \
    -DCMAKE_BUILD_TYPE="$build_type"

  # --config needed for windows
  cmake --build "$build_dir" --config "$build_type"
done
