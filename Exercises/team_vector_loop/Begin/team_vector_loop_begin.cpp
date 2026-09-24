// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project


// EXERCISE 6 Goal:
//   Convert to three-level team parallelism using team policy within the nested loops.

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

void checkSizes( int &M, int &N, int &S, int &E, int &nrepeat );

int main( int argc, char* argv[] )
{
  int M = -1;         // number of rows 2^8
  int N = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
  int E = -1;         // number of Elements
  int nrepeat = 100;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      M = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User M is %d\n", M );
    }
    else if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      N = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
      S = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User S is %d\n", S );
    }
    else if ( ( strcmp( argv[ i ], "-E" ) == 0 ) || ( strcmp( argv[ i ], "-Elements" ) == 0 ) ) {
      E = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User E is %d\n", E );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-M) <int>:      exponent num, determines number of rows 2^num (default: 2^8 = 256)\n" );
      printf( "  -Columns (-N) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^18 = 256*1024 )\n" );
      printf( "  -Elements (-E) <int>:  exponent num, determines number of elements 2^num (default: 2^10 = 1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( M, N, S, E, nrepeat );

  Kokkos::initialize( argc, argv );
  {

  using Layout = Kokkos::LayoutRight;

  using range_policy = Kokkos::RangePolicy<>;

  // Allocate y, x vectors and Matrix A on device.
  using ViewVectorType = Kokkos::View<double**, Layout>;
  using ViewMatrixType = Kokkos::View<double***, Layout>;
  ViewVectorType y( "y", E, M );
  ViewVectorType x( "x", E, N );
  ViewMatrixType A( "A", E, M, N );

  // Create host mirrors of device views.
  auto h_y = Kokkos::create_mirror_view( y );
  auto h_x = Kokkos::create_mirror_view( x );
  auto h_A = Kokkos::create_mirror_view( A );

  for ( int e = 0; e < E; e++ ) {
    // Initialize y vector on host.
    for ( int i = 0; i < M; ++i ) {
      h_y( e, i ) = 1;
    }

    // Initialize x vector on host.
    for ( int i = 0; i < N; ++i ) {
      h_x( e, i ) = 1;
    }

    // Initialize A matrix on host.
    for ( int j = 0; j < M; ++j ) {
      for ( int i = 0; i < N; ++i ) {
        h_A( e, j, i ) = 1;
      }
    }
  }

  // Deep copy host views to device views.
  Kokkos::deep_copy( y, h_y );
  Kokkos::deep_copy( x, h_x );
  Kokkos::deep_copy( A, h_A );

  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = Kokkos::TeamPolicy<>::member_type;

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    Kokkos::parallel_reduce( "yAx", team_policy( E, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember, double &update ) {
      const int e = teamMember.league_rank();

      // EXERCISE: Replace reduction over M with TeamThread parallelism.
      for ( int j = 0; j < M; j++ ) {
        double tempN = 0;

        // EXERCISE: Replace TeamThread parallelism with Vector parallelism.
        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int i, double &innerUpdateN ) {
          innerUpdateN += A( e, j, i ) * x( e, i );
        }, tempN );

        // EXERCISE: Replace team_rank check with Kokkos::single construct.
        if ( teamMember.team_rank() == 0 ) update += y( e, j ) * tempN;
      }
    }, result );

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d x %d is %lf\n", M, N, E, result );
    }

    const double solution = (double) M *(double) N *(double) E;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // The following is performed for each of E elements.
  //   Each matrix A row (each of length N) is read once.
  //   The x vector (of length N) is read M times.
  //   The y vector (of length M) is read once.
  // double Gbytes = 1.0e-9 * E * double( sizeof(double) * ( 2 * N * M + M ) );
  double Gbytes = 1.0e-9 * E * double( sizeof(double) * ( N + N * M + M ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  M( %d ) N( %d ) E( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          M, N, E, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  }
  Kokkos::finalize();

  return 0;
}

void checkSizes( int &M, int &N, int &S, int &E, int &nrepeat ) {
  // If S is undefined and M or N is undefined, set S to 2^18 or the bigger of M and N.
  if ( S == -1 && ( M == -1 || N == -1 ) ) {
    S = pow( 2, 18 );
    if ( S < M ) S = M;
    if ( S < N ) S = N;
  }

  // If S is undefined and both M and N are defined, set S = M * N.
  if ( S == -1 ) S = M * N;

  // If both M and N are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if ( M == -1 && N == -1 ) {
    if ( S > 1024 ) {
      N = 1024;
    }
    else {
      N = S;
    }
  }

  // If only N is undefined, set it.
  if ( N == -1 ) N = S / M;

  // If M is undefined, set it.
  if ( M == -1 ) M = S / N;

  // If E is undefined, set it to 2^10 = 1024.
  if ( E == -1 ) E = pow( 2, 10 );

  printf( "  Total size S = %d M = %d N = %d E = %d\n", S, M, N, E );

  // Check sizes.
  if ( ( S < 0 ) || ( M < 0 ) || ( N < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( M * N ) != S ) {
    printf( "  M * N != S\n" );
    exit( 1 );
  }
}
