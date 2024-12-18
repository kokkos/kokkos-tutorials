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

All the tutorial folders can be built using either the `Makefile` or the CMake `CMakeLists.txt` file in each folder.

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

## Makefiles

The raw Makefiles require Makefile variables to be properly configured. 
In most examples, this is `KOKKOS_PATH` pointing to the Kokkos source directory
and `KOKKOS_DEVICES` which contains the list of device backends to build.
This will build a new Kokkos library for each exercise.

If you are on a system compatible to our AWS instances, you can type 
```shell
make
make test
```
in the `Exercises` directory.

Compatible means:
 * X86 with a NVIDIA V100 GPU
 * kokkos was cloned to ${HOME}/Kokkos/kokkos

# Using `spack` to build Kokkos and Kokkos-Kernels

[`spack`](https://spack.io) is a package manager for supercomputers, Linux, and macOS. It makes installing scientific
software easy. With `spack`, you can build Kokkos and its dependencies with a single command.

We provide environments to ease the setup of Kokkos and Kokkos-Kernels for the tutorials. The environments are located
in the `spack-envs` directory.

To use the environments, first install `spack` by following the
instructions [here](https://spack.readthedocs.io/en/latest/getting_started.html).

To build Kokkos and Kokkos-Kernels with OpenMP backend, you can use the following commands:

```bash
source <spack_install_dir>/share/spack/setup-env.sh
spack compiler add  # find compilers on your system
spack env activate spack-envs/kokkos-kernels/openmp
spack install
spack load kokkos-kernels
```

Now you can go to the tutorial directory and build the tutorials using the installed Kokkos-Kernels library.

For example:

```bash
# Select the exercise, and cheat to the solution directory
cd Exercises/kokkoskernels/GaussSeidel/Solution
# Configure the build
cmake -B build_dir
# Build the exercise
cmake --build build_dir
# Run the exercise
./build_dir/gauss_seidel
```
