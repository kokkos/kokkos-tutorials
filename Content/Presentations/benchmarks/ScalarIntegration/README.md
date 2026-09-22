# Scalar-integration benchmark

This benchmark generates the data for the `Amdahl's Law (2)` figure in Module
1. It measures midpoint integration of `4 / (1 + x*x)` with a serial loop and
with `Kokkos::parallel_reduce`. Each reported time is the median of 20 samples.

## Build

Configure against the same Kokkos installation used for the tutorials. For an
H200 CUDA build, for example:

```console
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build --parallel
```

## Run

Use a descriptive label containing the accelerator or CPU model and backend.
Standard output is CSV data; standard error records the Kokkos configuration.

```console
./build/scalar_integration --label "NVIDIA H200 (CUDA)" \
  > h200-cuda.csv 2> h200-cuda-metadata.txt
```

If the same Kokkos build enables OpenMP, collect the host CPU curve from the
same node with:

```console
export OMP_NUM_THREADS=<physical-core-count>
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
./build/scalar_integration --host-execution-space \
  --label "<CPU model> (OpenMP)" \
  > cpu-openmp.csv 2> cpu-openmp-metadata.txt
```

For OpenMP runs, also record `OMP_NUM_THREADS`, `OMP_PROC_BIND`, and
`OMP_PLACES` in the metadata file. Record the compiler version, Kokkos version
or commit, node model, and GPU model with every result. Run on an otherwise idle
compute node using the site's normal job launcher.

The serial reference is measured on the host CPU of each tested node. Therefore
each series represents the speedup delivered by that node's configured Kokkos
execution space over a serial loop on the same node.

## Plot

Pass all result files to the plotting script and replace the slide figure:

```console
python3 plot.py results/a100-cuda.csv \
  --output ../../modularized/figures/ScalarIntegration.pdf
```

Commit the CSV files and their matching metadata files along with the generated
PDF so that the plot remains auditable and reproducible.
