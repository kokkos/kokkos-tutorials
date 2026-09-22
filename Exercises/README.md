| Exercise name | Information |
| --- | --- |
| 01 | This exercise involves converting the loops in the given code to parallel constructs using the Kokkos library. |
| 02 | This exercise aims to replace memory allocations with Kokkos Views in the provided code. |
| 03 | In this exercise, the code expands on the previous exercise by introducing the concept of Kokkos mirrors. Kokkos mirrors allow for synchronization and data transfer between the host and device memory spaces. |
| 04 | In this exercise, the code introduces additional features and customization options for the Kokkos execution space, memory space, layout, and range policy. |
| dualview | This exercise demonstrates the use of DualView to manage data and computations that take place on two different memory spaces, such as device memory and host memory. |
| fortran-kokkosinterface |
| hpcbind | This exercise demonstrates the use of the Hardware Locality (hwloc) library and OpenMP to determine the binding of threads to CPU cores and processing units (PUs). |
| instances | The exercise in the code introduces the use of instances in Kokkos. Instances allow you to partition the execution space into multiple subsets and execute parallel operations concurrently on each subset. |
| mdrange | This exercise demonstrates the use of parallelize matrix-vector multiplication and dot product calculations using Kokkos' parallel patterns. |
| mpi\_exch | This exercise demonstrates how to perform data exchange between MPI ranks using non-blocking communication operations |
| mpi\_heat\_conduction | This exercise is a parallel simulation of a heat transfer problem using Kokkos and MPI. |
| mpi\_pack\_unpack | The purpose of this exercise is to demonstrate how to use MPI (Message Passing Interface) with Kokkos |
| random\_number | This exercise showcases the usage of Kokkos' random number generator and how to perform parallel reduction to count hits within a circular region. The exercise also explores the impact of different parameters, such as the number of darts thrown and the generator type, on the accuracy of the pi estimation. |
| scatter\_view | This exercise demonstrates the use of different parallelization strategies, namely atomic updates and data replication, for performing a scatter add operation. |
| simd | The purpose of this exercise is to compare the performance of scalar computations and SIMD computations using the Kokkos library for a given problem size and number of iterations. |
| simp\_warp | This exercise compares the performance of SIMD (Single Instruction, Multiple Data) operations and team-vector operations. |
| subview | The purpose of this exercise is to demonstrate and practice using the Kokkos library to perform matrix-vector multiplication on different execution spaces (e.g., serial, threads, OpenMP, CUDA) with various memory spaces (e.g., host, device, CUDA unified memory). |
| tasking | The purpose of this exercise is to convert the serial Fibonacci code into a task-parallel version using the Kokkos library. |
| team\_policy | The purpose of this exercise is to convert a given code that performs matrix-vector multiplication into a team parallel implementation using the Kokkos library. |
| team\_sratch\_memory | The purpose of this exercise is to utilize scratch memory to explicitly cache the x vector in the matrix-vector multiplication code. The goal is to improve performance by reducing memory accesses and taking advantage of data locality. |
| team\_vector\_loop | The purpose of this exercise is to convert the existing code to three-level team parallelism using the team policy within the nested loops. |
| tools\_minind |
| unique\_token | The purpose of the exercise is to modify the given code to utilize Kokkos' token-based team parallelism and implement a scatter-add algorithm using data replication |
| unordered\_map | The purpose of this exercise is to practice using Kokkos' UnorderedMap container and perform operations on it. |
| vectorshift | The goal of this exercise is to learn how to use Partitioned Global Address Space (PGAS) to implement a circular vector shift. |
| virtualfunction | In this exercise, the goal is to launch a parallel kernel to create virtual objects on the device using placement new, and then another parallel kernel to destroy those objects before freeing the memory. |
