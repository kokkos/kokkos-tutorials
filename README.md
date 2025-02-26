![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)

# Kokkos Tutorials

This repository contains tutorials for the Kokkos C++ programming
model (github.com/kokkos/kokkos). 

## The Kokkos Lectures

We are currently running **The Kokkos Lectures** - an extended version
of our **Intro-Full** Tutorial, spanning 8 modules. For information on that, 
Slides and Recordings visit: [The Kokkos Lectures Wiki](https://github.com/kokkos/kokkos-tutorials/wiki/Kokkos-Lecture-Series)

## Other Tutorial Compilations

Tutorials in the **Intro-Short** directory cover
 * simple data parallel patterns and policies
 * multidimensional array views
 * execution and memory spaces

Tutorials in the **Intro-Full** directory cover
 * simple data parallel patterns and policies
 * multidimensional array views
 * execution and memory spaces
 * thread safety and atomic operations
 * hierarchical data parallel patterns

# Building the Tutorials

All the tutorial folders can be built using CMake.

## CMake

CMake can build against an installed Kokkos library or download one automatically using `FetchContent`.

You can use the global `CMakeLists.txt` at the top level directory to build all the exercises simultaneously. Else, you
can run the very same command in each exercise directory to build them only one by one.

```shell
cmake -B build_dir
cmake --build build_dir
```

A specific CMake option, `KokkosTutorials_FORCE_INTERNAL_Kokkos`, can be used to force the use of the internal Kokkos
library and only use an already installed one that can be too old or not configured as wished. An opposite
option, `KokkosTutorials_FORCE_EXTERNAL_Kokkos` can prevent Kokkos from being downloaded.

```shell
# Download and build Kokkos and the tutorials
cmake -B build_dir -DKokkosTutorials_FORCE_INTERNAL_Kokkos=ON # -DKokkos_* options
cmake --build build_dir
```

Kokkos setup is covered by the [quickstart guide](https://kokkos.org/kokkos-core-wiki/quick_start.html) and an exhaustive
Kokkos options are described in [CMake options](https://kokkos.org/kokkos-core-wiki/keywords.html).

For specific use-cases, like when an internet connection is not available, the `KokkosTutorials_KOKKOS_SOURCE_DIR` can
be used to point to a local Kokkos source directory.

Here are some examples of building the exercises with CMake:

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

