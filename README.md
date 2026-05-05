![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)

# Kokkos Tutorials

This repository contains tutorials for the Kokkos C++ programming
model (github.com/kokkos/kokkos). 

## The Kokkos Lectures

The most comprehensive recording of our tutorials - the **The Kokkos Lectures** - are available on YouTube.
For more information on that, including the
Slides and Recordings visit: [The Kokkos Lectures Wiki](https://github.com/kokkos/kokkos-tutorials/wiki/Kokkos-Lecture-Series)

## Old Slides

The old slides are available in this repository under the [legacy tag](https://github.com/kokkos/kokkos-tutorials/tree/legacy).

# Building the Tutorials

All the tutorial folders can be built using CMake.

## CMake

CMake can build against an installed Kokkos library or download one automatically using `FetchContent`.

Without any Kokkos already installed, from an exercise directory, one can run the following:

```shell
cmake -B build_dir # -DKokkos_* options
cmake --build build_dir
```

Kokkos options are described in [CMake options](https://kokkos.org/kokkos-core-wiki/keywords.html).

For example, OpenMP CPU exercises can be built as:
```shell
cmake -B build_openmp -DKokkos_ENABLE_OPENMP=ON
cmake --build build_openmp
```

On Mac, if OpenMP is not available, one can use the Threads backend:
```shell
cmake -B build_threads -DKokkos_ENABLE_THREADS=ON
cmake --build build_threads
```

For a NVIDIA gpu, using gpu arch autodetection:

```shell
cmake -B build_cuda -DKokkos_ENABLE_CUDA=ON
cmake --build build_cuda
```

To pass an already installed Kokkos library, you can use classical CMake variables,
such as `Kokkos_ROOT`, or `CMAKE_PREFIX_PATH`.

