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

#include <Kokkos_Core.hpp>

#ifndef KOKKOS_ENABLE_CUDA
#error "This exercise can only be run with Kokkos_ENABLE_CUDA=ON"
#else

using HostSpace      = Kokkos::HostSpace;
using ExecSpace      = Kokkos::DefaultExecutionSpace;
using TeamPolicy     = Kokkos::TeamPolicy<ExecSpace>;
using MemberType     = TeamPolicy::member_type;
using ViewVectorType = Kokkos::View<double*>;
using ViewMatrixType = Kokkos::View<double**>;

// Use a view result type for parallel reduce to avoid fencing over all
// execution spaces. Without this the kernels will run serially.
using ResultType = Kokkos::View<double>;

struct CudaStreams {
  std::array<int, 2> devices;
  std::array<cudaStream_t, 2> streams;

  CudaStreams() {
    // Query number of devices available
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    // Choose 2 devices for this tutorial
    devices = {0, n_devices - 1};

    for (auto i = 0; i < devices.size(); ++i) {
      // Set device for Cuda API calls
      cudaSetDevice(devices[i]);

      // Create Cuda stream
      cudaStreamCreate(&streams[i]);
    }
  }

  ~CudaStreams() {
    for (auto i = 0; i < devices.size(); ++i) {
      // Set device for Cuda API calls
      cudaSetDevice(devices[i]);

      // Destroy Cuda stream
      cudaStreamDestroy(streams[i]);
    }
  }

  // Removing the following ensure that we manage the lifetime of the streams
  CudaStreams(const CudaStreams &) = delete;
  CudaStreams &operator=(const CudaStreams &) = delete;
};

void operation(ExecSpace& exec_space, ResultType& result, ViewMatrixType& A,
               ViewVectorType& y, ViewVectorType& x) {
  // Application: <y, Ax> = y^T*A*x
  const int N = x.extent(0);

  // Use execution space for deep_copy to correct device
  Kokkos::deep_copy(exec_space, y, 1.0);
  Kokkos::deep_copy(exec_space, x, 1.0);
  Kokkos::deep_copy(exec_space, A, 1.0);

  // Pass execution space to policy constructor to launch on correct device
  auto policy = TeamPolicy(exec_space, N, Kokkos::AUTO, 32);
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

  // Create Cuda streams
  CudaStreams cuda_streams;

  Kokkos::initialize(argc, argv);
  {
    // We scope the creation of all execution spaces and views to ensure
    // they are destroyed before the cuda streams themselves are destroyed.

    // Use streams to construct execution space on different devices.
    std::array<ExecSpace, 2> execs = {ExecSpace(cuda_streams.streams[0]),
                                      ExecSpace(cuda_streams.streams[1])};

    // Allocate views for use on different devices
    ViewVectorType y0(Kokkos::view_alloc("y0", execs[0]), N);
    ViewVectorType x0(Kokkos::view_alloc("x0", execs[0]), N);
    ViewMatrixType A0(Kokkos::view_alloc("A0", execs[0]), N, N);

    ViewVectorType y1(Kokkos::view_alloc("y1", execs[1]), N);
    ViewVectorType x1(Kokkos::view_alloc("x1", execs[1]), N);
    ViewMatrixType A1(Kokkos::view_alloc("A1", execs[1]), N, N);

    // Timer
    Kokkos::Timer timer;

    // Allocate result views for use on different devices
    ResultType result0(Kokkos::view_alloc("result0", execs[0]));
    ResultType result1(Kokkos::view_alloc("result1", execs[1]));
    for (int repeat = 0; repeat < nrepeat; repeat++) {
      // Pass execution space instances for deep copying and launching
      // kernels on different devices
      operation(execs[0], result0, A0, y0, x0);
      operation(execs[1], result1, A1, y1, x1);

      // Get results on host
      auto result0_h =
          Kokkos::create_mirror_view_and_copy(HostSpace(), result0);
      auto result1_h =
          Kokkos::create_mirror_view_and_copy(HostSpace(), result1);

      // Check results
      const double solution = (double)N * (double)N;
      if (result0_h() != solution) {
        printf("  Error: result0(%e) != solution(%e)\n", result0_h(),
                solution);
      }
      if (result1_h() != solution) {
        printf("  Error: result1(%e) != solution(%e)\n", result1_h(),
                solution);
      }

      // Output results
      if (repeat == (nrepeat - 1)) {
        Kokkos::fence();
        printf("  Computed results for N=%d and nrepeat=%d are %e and %e\n",
                N, nrepeat, result0_h(), result1_h());
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
