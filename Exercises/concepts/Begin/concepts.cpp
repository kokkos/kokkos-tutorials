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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include <Kokkos_Core.hpp>

template<class V>
concept AnyView = Kokkos::is_view_v<V>;

void checkSizes( int &N, int &M, int &S, int &nrepeat );

// EXERCISE 1: use AnyView and Kokkos::ExecutionSpace concepts for the functions
// EXERCISE 2: rename init_vector and init_matrix to init_view, and use constraints
//             (requires clause) to distinguish the overloads
// EXERCISE 3: Can you do the init_view version as a single function with if constexpr?
//             What is the downside of that?
template<class Exec, class Vector>
void init_vector(Exec exec, Vector x, typename Vector::element_type val) {
  Kokkos::parallel_for("init_vector", Kokkos::RangePolicy(exec, 0, x.extent_int(0)),
    KOKKOS_LAMBDA(int i) { x(i) = val; });
}

template<class Exec, class Matrix>
void init_matrix(Exec exec, Matrix A, typename Matrix::element_type val) {
  Kokkos::parallel_for("init_vector",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, Exec>(exec, {0,0}, {A.extent_int(0), A.extent_int(1)}),
    KOKKOS_LAMBDA(int i, int j) { A(i,j) = val; });
}

// EXERCISE: constrain the function to Views that have arithmetic types as
//           element types (using requires clause, together with the AnyView concept)
// EXERCISE: Can you write a Matrix and Vector concept instead?
// EXERCISE: where and how should you check at compiler time that compile time
//           extents match where appropriate (instead of just runtime lengths)
//           Remember static_extent(r) == Kokkos::dynamic_extent for dynamic extents
template<class Exec, class Matrix, class VectorX, class VectorY>
double run_code(Exec exec, Matrix A, VectorX x, VectorY y) {  
  double result;
  int N = A.extent(0);
  int M = A.extent(1);
  if(x.extent(0) != M) Kokkos::abort("X length mismatches number of columns in A");
  if(y.extent(0) != N) Kokkos::abort("Y length mismatches number of rows in A");
  Kokkos::parallel_reduce( "yAx", Kokkos::RangePolicy(exec, 0, N),
    KOKKOS_LAMBDA ( int j, double &update ) {
      double temp2 = 0;

      for ( int i = 0; i < M; ++i ) {
        temp2 += A( j, i ) * x( i );
      }

      update += y( j ) * temp2;
    }, result );
  return result;
}

int main( int argc, char* argv[] )
{
  int N = -1;         // number of rows 2^12
  int M = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
  int nrepeat = 100;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User M is %d\n", M );
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
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, M, S, nrepeat );

  Kokkos::initialize( argc, argv );
  {

  // Allocate y, x vectors and Matrix A on device.
  using ViewVectorType = Kokkos::View<double*, Kokkos::SharedSpace>;
  using ViewMatrixType = Kokkos::View<double**, Kokkos::SharedSpace>;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", M );
  ViewMatrixType A( "A", N, M );

  auto exec = Kokkos::DefaultExecutionSpace();

  // Initialize y vector on host.
  init_vector(exec, y, 1);

  // Initialize x vector on host.
  init_vector(exec, x, 1);

  // Initialize A matrix on host, note 2D indexing.
  init_matrix(exec, A, 1);
  Kokkos::fence();

  // EXERCISE: Check error message when using the wrong
  // function call here before and after introducing concepts
  // init_vector(exec, A, 1);
  // init_vector(A, exec, 1);

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = run_code(exec, A, x, y);

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
