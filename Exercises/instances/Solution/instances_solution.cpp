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

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

using ExecSpace = Kokkos::DefaultExecutionSpace;
using range_policy = Kokkos::RangePolicy<ExecSpace>;
using ViewVectorType = Kokkos::View<double*>;
using ViewMatrixType = Kokkos::View<double**>;

#ifdef KOKKOS_ENABLE_CUDA
#define RESULT_MEM_SPACE Kokkos::CudaHostPinnedSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
#define RESULT_MEM_SPACE Kokkos::HIPHostPinnedSpace
#endif
#ifndef RESULT_MEM_SPACE
#define RESULT_MEM_SPACE Kokkos::HostSpace
#endif
using ResultType = Kokkos::View<double, RESULT_MEM_SPACE>;

void operation(ExecSpace instance, ResultType result, ViewMatrixType A, ViewVectorType y, ViewVectorType x) {
  int N = x.extent(0);
  Kokkos::deep_copy(instance, y, -2.5);
  Kokkos::parallel_for("VectorAdd", range_policy(instance, 0,N), KOKKOS_LAMBDA(int i) {
    x(i) += y(i);
  });
  Kokkos::parallel_for("MatVec", range_policy(instance, 0,N), KOKKOS_LAMBDA(int i) {
    double tmp = 0;
    for(int j=0; j<N; j++) {
      tmp += A(i,j)*x(j);
    }
    y(i) = tmp;
  });
  Kokkos::parallel_reduce("Dot", range_policy(instance, 0,N), KOKKOS_LAMBDA(int i, double& lsum) {
    lsum += y(i)*y(i);
  },result);
}

int main( int argc, char* argv[] )
{
  int64_t N = 10000;     // number of rows/cols
  int nrepeat = 10;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = atoi( argv[ ++i ] );
      printf( "  User N is %d\n", N );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  -Rows (-N) <int>:      number of rows and colums (default: 20000)\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }


  Kokkos::initialize( argc, argv );
  {

  ViewVectorType y1( "y1", N );
  ViewVectorType x1( "x1", N );
  ViewMatrixType A1( "A1", N, N );
  Kokkos::deep_copy( y1, 2.0 );
  Kokkos::deep_copy( x1, 3.0 );
  Kokkos::deep_copy( A1, 4.0 );

  ViewVectorType y2( "y2", N );
  ViewVectorType x2( "x2", N );
  ViewMatrixType A2( "A2", N, N );
  Kokkos::deep_copy( y2, 2.5 );
  Kokkos::deep_copy( x2, 3.5 );
  Kokkos::deep_copy( A2, 4.5 );

  auto instances = Kokkos::Experimental::partition_space(ExecSpace(), 1, 1);
  // Timer products.
  Kokkos::Timer timer;

  ResultType result1("R1"), result2("R2");
  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    operation(instances[0], result1, A1, y1, x1);
    operation(instances[1], result2, A2, y2, x2);

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      Kokkos::fence();
      printf( "  Computed results for %ld and %d are %e and %e\n", N, nrepeat, result1(), result2() );
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  double Gbytes = 2.0e-9 * double( sizeof(double) * ( 6. * N + 2. * N * N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %ld ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, nrepeat, 1.e-6*(N*N+2*N)*sizeof(double), time, Gbytes * nrepeat / time );

  }
  Kokkos::finalize();

  return 0;
}

