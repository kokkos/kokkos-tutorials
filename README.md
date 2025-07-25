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

All the tutorials are built using CMake.

## CMake Quickstart

From the top level directory or from any exercise directory, you can build the tutorials using CMake:

```shell
cmake -B build_dir
cmake --build build_dir
```

Additional options can be passed to CMake to configure the build, such as the backend to use, the architecture to target, etc.

## Examples

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

For an AMD GPU with autodetection of the GPU architecture:

```shell
cmake -B build_hip -DKokkos_ENABLE_HIP=ON
cmake --build build_hip
```

Kokkos setup is covered by the [quickstart guide](https://kokkos.org/kokkos-core-wiki/get-started/quick-start.html) and an exhaustive list of Kokkos options is detailed in the [CMake keywords documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).

## Advanced CMake Usage

CMake can build against an existing Kokkos installation or download the source files automatically using `FetchContent`.

To pass an already installed Kokkos library, you can use classical CMake variables,
such as `Kokkos_ROOT`, or `CMAKE_PREFIX_PATH`.

A specific CMake option, `CMAKE_DISABLE_FIND_PACKAGE_Kokkos`, can be used to force the use of the internal Kokkos
library, discarding any already installed Kokkos.

An opposite option, `CMAKE_REQUIRE_FIND_PACKAGE_Kokkos` can prevent Kokkos from being downloaded and is useful to
test against an already installed Kokkos.

```shell
# Download and build Kokkos and the tutorials, forcing the use of the internal Kokkos
cmake -B build_dir -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON # -DKokkos_* options
cmake --build build_dir
```

For specific use-cases, like when an internet connection is not available, the `KokkosTutorials_KOKKOS_SOURCE_DIR` can
be used to point to a local Kokkos source directory.
For example,

```shell
cmake -B build_dir -DKokkos_ENABLE_THREADS=ON  -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON \
 -DKokkosTutorials_KOKKOS_SOURCE_DIR=<PATH_TO_KOKKOS_SOURCE>
```
