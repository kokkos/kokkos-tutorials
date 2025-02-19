// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include<Kokkos_Core.hpp>
#include<Kokkos_SIMD.hpp>

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

  using simd_t = Kokkos::Experimental::simd<double>;

  // Create simd views.
  Kokkos::View<simd_t*, Layout, MemSpace> x( "x", M_simd);;
  Kokkos::View<simd_t**, Layout, MemSpace> A( "A", N, M_simd);  

  // Unmanaged views for initialization.
  Kokkos::View<double*, Layout, MemSpace> x_scalar((double*)x.data(), M);
  Kokkos::View<double**, Layout, MemSpace> A_scalar((double*)A.data(), N, M);
  Kokkos::View<double*, Layout, MemSpace> y( "y", N);

  // Initialize y vector.
  for ( int i = 0; i < N; ++i ) {
    y( i ) = 1;
  }
  // Initialize x vector.
  for ( int i = 0; i < M; ++i ) {
    x_scalar( i ) = 1;
  }
  // Initialize A matrix.
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < M; ++i ) {
      A_scalar( j, i ) = 1;
    }
  }

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    Kokkos::parallel_reduce( "yAx", range_policy( 0, N ), KOKKOS_LAMBDA ( int j, double &update ) {
      simd_t temp2(0.0);
      for ( int i = 0; i < M_simd; ++i ) {
        temp2 += A( j, i ) * x( i );
      }       
      // Accumulate the result for each SIMD lane.
      for ( std::size_t i = 0; i < simd_t::size(); ++i) {
        update += y(j) * temp2[i];
      }
    }, result );
    
    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d is %lf\n", N, M, result );
    }

    const double solution = (double) N * (double) M;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  }
  Kokkos::finalize();

  return 0;
}

void checkSizes( int &N, int &M, int &S, int &nrepeat ) {
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if ( S == -1 && ( N == -1 || M == -1 ) ) {
    S = pow( 2, 22 );
    if ( S < N ) S = N;
    if ( S < M ) S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if ( S == -1 ) S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if ( N == -1 && M == -1 ) {
    if ( S > 1024 ) {
      M = 1024;
    }
    else {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = S / N;

  // If N is undefined, set it.
  if ( N == -1 ) N = S / M;

  printf( "  Total size S = %d N = %d M = %d\n", S, N, M );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }
}
