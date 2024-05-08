//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// EXERCISE Goal:
//   Launch kernels on multiple GPU devices which execute simultaneously

#include <Kokkos_Core.hpp>

#ifndef KOKKOS_ENABLE_CUDA
#error "This exercise can only be run with Kokkos_ENABLE_CUDA=ON"
#else

using ExecSpace      = Kokkos::DefaultExecutionSpace;
using TeamPolicy     = Kokkos::TeamPolicy<ExecSpace>;
using MemberType     = TeamPolicy::member_type;
using ViewVectorType = Kokkos::View<double*>;
using ViewMatrixType = Kokkos::View<double**>;

// EXERCISE: choose a ResultType for parallel_reduce() that will not fence
using ResultType = double;

struct CudaStreams {
  std::array<int, 2> devices;
  std::array<cudaStream_t, 2> streams;

  CudaStreams() {
    // EXERCISE: query total number of devices available and choose 2: devices = {devid0, devid1}
    //             - Use cudaGetDeviceCount()
    // EXERCISE: create a stream on each chosen device
    //             - Use cudaSetDevice() to direct Cuda API to use particular device
    //             - Create stream using cudaStreamCreate()
  }

  ~CudaStreams() {
    // EXERCISE: for each stream, destroy on the correct device
    //             - Use cudaSetDevice() to direct Cuda API to use particular device
    //             - Destroy stream using cudaStreamDestroy()
  }

  // Removing the following ensure that we manage the lifetime of the streams
  CudaStreams(const CudaStreams &) = delete;
  CudaStreams &operator=(const CudaStreams &) = delete;
};

// EXERCISE: pass in an execution space instance
void operation(ResultType& result, ViewMatrixType& A, ViewVectorType& y,
               ViewVectorType& x) {
  // Application: <y, Ax> = y^T*A*x
  const int N = x.extent(0);

  // EXERCISE: deep copy on the correct device
  Kokkos::deep_copy(y, 1.0);
  Kokkos::deep_copy(x, 1.0);
  Kokkos::deep_copy(A, 1.0);

  // EXERCISE: launch team policy on the correct device
  auto policy = TeamPolicy(N, Kokkos::AUTO, 32);
  Kokkos::parallel_reduce(
      "y^TAx", policy,
      KOKKOS_LAMBDA(const MemberType& team, double& update) {
        const int j = team.league_rank();

        double temp = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team, N),
            [&](const int i, double& innerUpdate) {
              innerUpdate += A(j, i) * x(i);
            },
            temp);

        Kokkos::single(Kokkos::PerTeam(team), [&]() { update += y(j) * temp; });
      },
      result);
}

int main(int argc, char* argv[]) {
  int N       = 10000;  // number of rows/cols
  int nrepeat = 100;    // number of repeats of the test

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0)) {
      N = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  -Rows (-N) <int>:      number of rows and colums (default: 10000)\n");
      printf("  -nrepeat <int>:        number of repetitions (default: 100)\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  // EXERCISE: create a CudaStreams object

  Kokkos::initialize(argc, argv);
  {
    // EXERCISE: create execution space instances using the streams in CudaStreams

    // EXERCISE: allocate on device 0
    ViewVectorType y0("y0", N);
    ViewVectorType x0("x0", N);
    ViewMatrixType A0("A0", N, N);

    // EXERCISE: allocate on device 1
    ViewVectorType y1("y1", N);
    ViewVectorType x1("x1", N);
    ViewMatrixType A1("A1", N, N);

    // Timer
    Kokkos::Timer timer;

    // EXERCISE: correctly allocate new ResultType
    ResultType result0;
    ResultType result1;
    for (int repeat = 0; repeat < nrepeat; repeat++) {
      // EXERCISE: pass an execution space instances
      operation(result0, A0, y0, x0);
      operation(result1, A1, y1, x1);

      // EXERCISE: process results correctly after changing result type

      // Check results
      const double solution = (double)N * (double)N;
      if (result0 != solution) {
        printf("  Error: result0(%e) != solution(%e)\n", result0, solution);
      }
      if (result1 != solution) {
        printf("  Error: result1(%e) != solution(%e)\n", result1, solution);
      }

      // Output result.
      if (repeat == (nrepeat - 1)) {
        Kokkos::fence();
        printf("  Computed results for N=%d and nrepeat=%d are %e and %e\n",
                N, nrepeat, result0, result1);
      }
    }

    // Calculate time.
    double time = timer.seconds();

    // Calculate bandwidth.
    double Gbytes = 2.0e-9 * double(sizeof(double) * (4. * N + 2. * N * N));

    // Print results (problem size, time and bandwidth in GB/s).
    printf("  N( %ld ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
            N, nrepeat, 1.e-6 * (2 * N * N + 4 * N) * sizeof(double), time,
            Gbytes * nrepeat / time);
  }
  Kokkos::finalize();

  return 0;
}

#endif
