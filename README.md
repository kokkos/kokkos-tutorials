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

## Makefiles

The raw Makefiles require Makefile variables to be properly configured. 
In most examples, this is `KOKKOS_PATH` pointing to the Kokkos source directory
and `KOKKOS_DEVICES` which contains the list of device backends to build.
This will build a new Kokkos library for each exercise.

If you are on a system compatible to our AWS instances, you can type 
```
make
make test
```
in the `Exercises` directory.

Compatible means:
 * X86 with a NVIDIA V100 GPU
 * kokkos was cloned to ${HOME}/Kokkos/kokkos

## CMake + Spack

The CMake files build against an installed Kokkos library. 
The easiest way to do this is using Spack.
There is a `spack.sh` script that automates most of this.
The Spack script can be run in any excercise folder with a `CMakeLists.txt`.

````
../../BuildScripts/spack.sh +openmp %gcc@7.3.0
````
This will make sure Kokkos is installed with the OpenMP backend for the GCC 7.3.0 compiler.
It will then configure CMake and create a `spack-build` folder where `make` can be run.
The `spack.sh` accepts the full list of variants and specs as the parent Kokkos package,
which can be viewed by running:

````
spack info kokkos
````

The `spack.sh` script uses the special DIY mode of Spack to install dependencies and configure the current source folder to build.

For Kokkos Kernels tutorials, there is similarly a `kk-spack.sh` script, e.g.
````
../../BuildScripts/kk-spack.sh +openmp %gcc@7.3.0 ^kokkos+aggressive_vectorization
````
All the arguments to the script get passed as a spec for the tutorial.
We are indirectly configuring Kokkos, hence the `^` notation for specifying the exact Kokkos dependency spec.


