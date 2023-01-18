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
//   - Implement inner product in two separate sub-exercises using:
//        Ex 1. KokkosKernels BLAS functions (gemv, dot)
//        Ex 2. KokkosKernels team-based BLAS functions using team parallelism with
//              team policy (team-based dot)
//   - Compare runtimes of these two implementations. Try different array layouts

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
// EXERCISE: Include header files for proper KokkosKernels BLAS functions.
// EXERCISE hint: KokkosBlas1_dot.hpp, KokkosBlas2_gemv.hpp, KokkosBlas1_team_dot.hpp
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas1_team_dot.hpp>

void checkSizes( int &N, int &M, int &S, int &nrepeat );

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
    // typedef Kokkos::DefaultExecutionSpace::array_layout  Layout;
    // typedef Kokkos::LayoutLeft   Layout;
    typedef Kokkos::LayoutRight  Layout;

    // Allocate y, x vectors and Matrix A on device.
    typedef Kokkos::View<double*, Layout>   ViewVectorType;
    typedef Kokkos::View<double**, Layout>  ViewMatrixType;
    ViewVectorType y( "y", N );
    ViewVectorType x( "x", M );
    ViewMatrixType A( "A", N, M );

    // Create host mirrors of device views.
    ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
    ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
    ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );

    // Initialize y vector on host.
    for ( int i = 0; i < N; ++i ) {
      h_y( i ) = 1;
    }

    // Initialize x vector on host.
    for ( int i = 0; i < M; ++i ) {
      h_x( i ) = 1;
    }

    // Initialize A matrix on host.
    for ( int j = 0; j < N; ++j ) {
      for ( int i = 0; i < M; ++i ) {
        h_A( j, i ) = 1;
      }
    }

    // Deep copy host views to device views.
    Kokkos::deep_copy( y, h_y );
    Kokkos::deep_copy( x, h_x );
    Kokkos::deep_copy( A, h_A );

    typedef Kokkos::TeamPolicy<>               team_policy;
    typedef Kokkos::TeamPolicy<>::member_type  member_type;

    // Timer products.
    struct timeval begin, end;

    //--------------------------------------------------------//
    //------------------       Ex. 1        ------------------//
    //------------------Using BLAS functions------------------//
    //--------------------------------------------------------//

    printf( "  Using BLAS functions: gemv, dot\n" );

    ViewVectorType tmp( "tmp", N );
    double alpha = 1;
    double beta  = 0;
	
    gettimeofday( &begin, NULL );

    for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
      // Application: <y,Ax> = y^T*A*x
      double result = 0;

      // EXERCISE: Convert from hierarchical parallel execution to using KokkosKernels BLAS functions
      // EXERCISE hint: KokkosBlas::gemv (tmp = A*x)
      //                KokkosBlas::dot (result = <y,tmp>)
      KokkosBlas::gemv("N",alpha,A,x,beta,tmp);
      result = KokkosBlas::dot(y,tmp);

      // Output result.
      if ( repeat == ( nrepeat - 1 ) ) {
        printf( "    Computed result for %d x %d is %lf\n", N, M, result );
      }

      const double solution = (double) N * (double) M;

      if ( result != solution ) {
        printf( "    Error: result( %lf ) != solution( %lf )\n", result, solution );
      }
    }

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );

    // Calculate bandwidth.
    // Each matrix A row (each of length M) is read once.
    // The x vector (of length M) is read N times.
    // The y vector (of length N) is read once.
    // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
    double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

    // Print results (problem size, time and bandwidth in GB/s).
    printf( "    N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
            N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

    
    //-------------------------------------------------------------------//
    //------------------           Ex. 2               ------------------//
    //------------------Using team-based BLAS functions------------------//
    //-------------------------------------------------------------------//

    printf( "  Using team-based dot\n" );
	
    ViewVectorType y2  ( "y2", N );
    ViewVectorType x2  ( "x2", M );
    ViewMatrixType A2  ( "A2", N, M );
    
    // Deep copy host views to device views.
    Kokkos::deep_copy( y2, h_y );
    Kokkos::deep_copy( x2, h_x );
    Kokkos::deep_copy( A2, h_A );
	
    gettimeofday( &begin, NULL );
    
    for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
      // Application: <y,Ax> = y^T*A*x
      double result2 = 0;
      const team_policy policy( N, Kokkos::AUTO );
      Kokkos::parallel_reduce( policy, KOKKOS_LAMBDA ( const member_type &teamMember, double &update ) {
        const int row = teamMember.league_rank();
    
        //EXERCISE: Multiply each row of matrix A2 with vector x2 using team-based dot functions
        //EXERCISE hint: - KokkosBlas::Experimental::dot (temp2 = <A2(row,:),x2>)
        //               - team-based dot, take subviews of View A2
        double temp2 = KokkosBlas::Experimental::dot(teamMember, Kokkos::subview(A2, row, Kokkos::ALL()), x2);
		
        if ( teamMember.team_rank() == 0 ) update += y2( row ) * temp2;
      }, result2 );
    
      // Output result.
      if ( repeat == ( nrepeat - 1 ) ) {
        printf( "    Computed result for %d x %d is %lf\n", N, M, result2 );
      }
    
      const double solution = (double) N * (double) M;
    
      if ( result2 != solution ) {
        printf( "    Error: result( %lf ) != solution( %lf )\n", result2, solution );
      }
    }
    
    gettimeofday( &end, NULL );
    
    // Calculate time.
    double time2 = 1.0 *   ( end.tv_sec - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );
    
    // Print results (problem size, time and bandwidth in GB/s).
    printf( "    N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
            N, M, nrepeat, Gbytes * 1000, time2, Gbytes * nrepeat / time2 );
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
