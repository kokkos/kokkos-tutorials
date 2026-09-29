// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project


// EXERCISE Goals:
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
    // typedef Kokkos::DefaultExecutionSpace::array_layout  Layout;
    // typedef Kokkos::LayoutLeft   Layout;
    typedef Kokkos::LayoutRight  Layout;

    // Allocate y, x vectors and Matrix A on device.
    typedef Kokkos::View<double*, Layout>   ViewVectorType;
    typedef Kokkos::View<double**, Layout>  ViewMatrixType;
    ViewVectorType y( "y", M );
    ViewVectorType x( "x", N );
    ViewMatrixType A( "A", M, N );

    // Create host mirrors of device views.
    auto h_y = Kokkos::create_mirror_view( y );
    auto h_x = Kokkos::create_mirror_view( x );
    auto h_A = Kokkos::create_mirror_view( A );

    // Initialize y vector on host.
    for ( int i = 0; i < M; ++i ) {
      h_y( i ) = 1;
    }

    // Initialize x vector on host.
    for ( int i = 0; i < N; ++i ) {
      h_x( i ) = 1;
    }

    // Initialize A matrix on host.
    for ( int j = 0; j < M; ++j ) {
      for ( int i = 0; i < N; ++i ) {
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

    // EXERCISE: Create ...
    gettimeofday( &begin, NULL );

    for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
      // Application: <y,Ax> = y^T*A*x
      double result = 0;

      // EXERCISE: Convert from hierarchical parallel execution to using KokkosKernels BLAS functions
      // EXERCISE hint: KokkosBlas::gemv (tmp = A*x)
      //                KokkosBlas::dot (result = <y,tmp>)
      Kokkos::parallel_reduce( team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember, double &update ) {
        const int j = teamMember.league_rank();
        double temp2 = 0;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int i, double &innerUpdate ) {
          innerUpdate += A( j, i ) * x( i );
        }, temp2 );

        if ( teamMember.team_rank() == 0 ) update += y( j ) * temp2;
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

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );

    // Calculate bandwidth.
    // Each matrix A row (each of length N) is read once.
    // The x vector (of length N) is read M times.
    // The y vector (of length M) is read once.
    // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * N * M + M ) );
    double Gbytes = 1.0e-9 * double( sizeof(double) * ( N + N * M + M ) );

    // Print results (problem size, time and bandwidth in GB/s).
    printf( "  M( %d ) N( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
            M, N, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

    
    //-------------------------------------------------------------------//
    //------------------           Ex. 2               ------------------//
    //------------------Using team-based BLAS functions------------------//
    //-------------------------------------------------------------------//

      ViewVectorType y2  ( "y2", M );
      ViewVectorType x2  ( "x2", N );
      ViewMatrixType A2  ( "A2", M, N );

      // Deep copy host views to device views.
      Kokkos::deep_copy( y2, h_y );
      Kokkos::deep_copy( x2, h_x );
      Kokkos::deep_copy( A2, h_A );

      gettimeofday( &begin, NULL );

      for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
        // Application: <y,Ax> = y^T*A*x
        double result2 = 0;
        const team_policy policy( M, Kokkos::AUTO );
        Kokkos::parallel_reduce( policy, KOKKOS_LAMBDA ( const member_type &teamMember, double &update ) {
          const int row = teamMember.league_rank();

          double temp2 = 0;
          //  EXERCISE: Multiply each row of matrix A2 with vector x2 using team-based dot functions; that is, replace
          //              the code in braces below (i.e. parallel_reduce with TeamThreadRange) with team-based dot
          //  EXERCISE  hint: - KokkosBlas::Experimental::dot (temp2 = <A2(row,:),x2>)
          //                  - team-based dot, take subviews of View A2
          {
            auto A2row = Kokkos::subview(A2, row, Kokkos::ALL());
            Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, A2row.extent(0)), [=] (int i, double &tmpUpdate) {
              tmpUpdate += A2row(i)*x2(i);
            }, temp2);
            teamMember.team_barrier();
          }

          if ( teamMember.team_rank() == 0 ) update += y2( row ) * temp2;
        }, result2 );

        // Output result.
        if ( repeat == ( nrepeat - 1 ) ) {
          printf( "    Computed result for %d x %d is %lf\n", M, N, result2 );
        }

        const double solution = (double) M * (double) N;

        if ( result2 != solution ) {
          printf( "    Error: result( %lf ) != solution( %lf )\n", result2, solution );
        }
      }

      gettimeofday( &end, NULL );

      // Calculate time.
      double time2 = 1.0 *   ( end.tv_sec - begin.tv_sec ) +
                    1.0e-6 * ( end.tv_usec - begin.tv_usec );

      // Print results (problem size, time and bandwidth in GB/s).
      printf( "    M( %d ) N( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
              M, N, nrepeat, Gbytes * 1000, time2, Gbytes * nrepeat / time2 );
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
