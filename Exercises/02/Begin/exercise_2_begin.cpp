// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project


// EXERCISE 2 Goal:
//   Replace raw allocations with Kokkos Views.
//     1. Define views in shared space.
//     2. Replace data access with view access operators.
//
//   Notes: * Kokkos::parallel_for() initializations were removed to initialize on host.
//          * Kokkos::SharedSpace allocates memory that is automatically migratable between host and device.

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include <Kokkos_Core.hpp>

void checkSizes( int &M, int &N, int &S, int &nrepeat );

int main( int argc, char* argv[] )
{
  int M = -1;         // number of rows 2^12
  int N = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
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
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-M) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-N) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( M, N, S, nrepeat );

  Kokkos::initialize( argc, argv );
  {

  // EXERCISE: Create views of the right size.

  // 1. Device Views
  // using ViewVectorType = Kokkos::View<double*, Kokkos::SharedSpace>;
  // using ViewMatrixType = Kokkos::View<double**, Kokkos::SharedSpace>;
  // ViewVectorType y( "y", M );
  // ViewVectorType x( "x", N );
  // ViewMatrixType A( "A", M, N );

  // EXERCISE: This no longer needs allocation after views introduced...
  //   Hint: If arrays are not allocated, they also do not need to be deallocated below
  // Allocate y, x vectors and Matrix A:
  double * const y = new double[ M ];
  double * const x = new double[ N ];
  double * const A = new double[ M * N ];

  // Initialize y vector on host.
  // EXERCISE: Convert y to 1D View's member access API: y(i)
  for ( int i = 0; i < M; ++i ) {
    y[ i ] = 1;
  }

  // Initialize x vector on host.
  // EXERCISE: Convert x to 1D View's member access API: x(i)
  for ( int i = 0; i < N; ++i ) {
    x[ i ] = 1;
  }

  // Initialize A matrix on host, note 2D indexing computation.
  // EXERCISE: convert 'A' to use View's member access API: A(j,i)
  for ( int j = 0; j < M; ++j ) {
    for ( int i = 0; i < N; ++i ) {
      A[ j * N + i ] = 1;
    }
  }

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    Kokkos::parallel_reduce( "yAx", M, KOKKOS_LAMBDA ( int j, double &update ) {
      double temp2 = 0;

      // EXERCISE: Replace access with view access operators.
      for ( int i = 0; i < N; ++i ) {
        temp2 += A[ j * N + i ] * x[ i ];
      }

      update += y[ j ] * temp2;
    }, result );

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d is %lf\n", M, N, result );
    }

    const double solution = (double) M * (double) N;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length N) is read once.
  // The x vector (of length N) is read M times.
  // The y vector (of length M) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * N * M + M ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( N + N * M + M ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  M( %d ) N( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          M, N, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  delete [] y;  //EXERCISE hint: ...
  delete [] x;  //EXERCISE hint: ...
  delete [] A;  //EXERCISE hint: ...

  }
  Kokkos::finalize();

  return 0;
}

void checkSizes( int &M, int &N, int &S, int &nrepeat ) {
  // If S is undefined and M or N is undefined, set S to 2^22 or the bigger of M and N.
  if ( S == -1 && ( M == -1 || N == -1 ) ) {
    S = pow( 2, 22 );
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

  printf( "  Total size S = %d M = %d N = %d\n", S, M, N );

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
