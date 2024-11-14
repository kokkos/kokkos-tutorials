#! /bin/bash

set -eou pipefail

kokkos_root="$1"
tutorials_src="$2"
cpp_compiler="$3"
build_type="$4"
backend="$5"

EXERCISES=(
  01
  02
  03
  04
)

export Kokkos_ROOT="$kokkos_root"
mkdir -p build
for e in "${EXERCISES[@]}"; do

  # CUDA hard-coded in, so CUDA needs to be enabled
  if [ "$e" == 04 ] && [ ! "$backend" == CUDA ]; then
    continue;
  fi

  for k in Begin Solution; do
    source_dir="$tutorials_src"/Exercises/"$e"/"$k"
    build_dir=build/"$source_dir"
    echo building "$source_dir"
    cmake -S "$source_dir" -B "$build_dir" \
      -DCMAKE_CXX_COMPILER="$cpp_compiler" \
      -DCMAKE_BUILD_TYPE="$build_type"
  
    # --config needed for windows
    cmake --build "$build_dir" --config "$build_type"
  done
done
