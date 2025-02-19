#! /bin/bash

set -eou pipefail

kokkos_root="$1"
kernels_root="$2"
tutorials_src="$3"
cpp_compiler="$4"
build_type="$5"
backend="$6"

# These are exercises with CMakeLists.txt in Begin and Solution subdirectories
# TODO: advanced_reductions seems broken
# TODO: hpcbind does not use cmake
# TODO: instances does not use cmake
# TODO: vectorshift needs Kokkos Remote Spaces
# TODO: kokkoskernels/CGSolve_SpILUKprecond needs to know where Kokkos Kernels source directory is
# TODO: kokkoskernels/SpILUK needs to know where Kokkos Kernels source directory is
# TODO: kokkoskernels/TeamGemm seems broken
# TODO: mpi_heat_conduction/no-mpi does not use cmake
BEGIN_SOLUTION_EXERCISES=(
01
02
03
dualview
kokkoskernels/BlockJacobi
kokkoskernels/GaussSeidel
kokkoskernels/GraphColoring
kokkoskernels/InnerProduct
mdrange
mpi_pack_unpack
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
  BEGIN_SOLUTION_EXERCISES+=(04)
  BEGIN_SOLUTION_EXERCISES+=(multi_gpu_cuda)
fi


if [ ! "$backend" == CUDA ]; then
  BEGIN_SOLUTION_EXERCISES+=(virtualfunction) # TODO: virtualfunction needs Kokkos with CUDA RDC
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
kokkoskernels/CGSolve/Solution # Begin does not include the proper headers (on purpose) so it can't be compiled
kokkoskernels/SpGEMM/Solution # Begin does not include the proper headers (on purpose) so it can't be compiled
mpi_exch
tools_minimd
)

# TODO: explicitly specifies CUDA
if [ "$backend" == CUDA ]; then
  EXERCISES+=(mpi_heat_conduction/Solution) # TODO: mpi_heat_conduction/Begin does not use cmake
fi

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
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DKokkosKernels_ROOT="$kernels_root"

  cmake --build "$build_dir"
done
