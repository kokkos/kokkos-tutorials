/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// EXERCISE 7 Goal:
//   Utilize scratch memory to explicitly cache the x vector.
//     1. Define a scratch view type.
//     2. Calculate the size of scratch space required.
//     3. Create a scratch view.
//     4. Fill the scratch view.
//     5. Use a barrier to make sure filling is done.
//     6. Use the scratch view.

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

void checkSizes( int &N, int &M, int &S, int &E, int &nrepeat );

int main( int argc, char* argv[] )
{
  int N = -1;         // number of rows 2^12
  int M = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
  int E = -1;         // number of Elements
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
    else if ( ( strcmp( argv[ i ], "-E" ) == 0 ) || ( strcmp( argv[ i ], "-Elements" ) == 0 ) ) {
      E = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User E is %d\n", E );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^8 = 256)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^18 = 256*1024 )\n" );
      printf( "  -Elements (-E) <int>:  exponent num, determines number of elements 2^num (default: 2^10 = 1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, M, S, E, nrepeat );

  Kokkos::initialize( argc, argv );
  {

  typedef Kokkos::LayoutRight  Layout;

  typedef Kokkos::RangePolicy<> range_policy;

  // Allocate y, x vectors and Matrix A on device.
  typedef Kokkos::View<double**, Layout>   ViewVectorType;
  typedef Kokkos::View<double***, Layout>  ViewMatrixType;

  // EXERCISE: Define Scratch Memory View Type.
  // typedef Kokkos::View<double*, ...., Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;

  ViewVectorType y( "y", E, N );
  ViewVectorType x( "x", E, M );
  ViewMatrixType A( "A", E, N, M );

  // Create host mirrors of device views.
  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );

  for ( int e = 0; e < E; e++ ) {
    // Initialize y vector on host.
    for ( int i = 0; i < N; ++i ) {
      h_y( e, i ) = 1;
    }

    // Initialize x vector on host.
    for ( int i = 0; i < M; ++i ) {
      h_x( e, i ) = 1;
    }

    // Initialize A matrix on host.
    for ( int j = 0; j < N; ++j ) {
      for ( int i = 0; i < M; ++i ) {
        h_A( e, j, i ) = 1;
      }
    }
  }

  // Deep copy host views to device views.
  Kokkos::deep_copy( y, h_y );
  Kokkos::deep_copy( x, h_x );
  Kokkos::deep_copy( A, h_A );

  typedef Kokkos::TeamPolicy<>               team_policy;
  typedef Kokkos::TeamPolicy<>::member_type  member_type;

  // EXERCISE: Calculate bytes per team for scratch space.
  // int scratch_size =  ScratchViewType:: ....

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    // EXERCISE: Tell policy how much scratch space is needed.
    Kokkos::parallel_reduce( team_policy( E, Kokkos::AUTO, 32 ), KOKKOS_LAMBDA ( const member_type &teamMember, double &update ) {
      const int e = teamMember.league_rank();
      double tempN = 0;

      // EXERCISE: Create a scratch view s_x for x.
      // ....

      if ( teamMember.team_rank() == 0 ) {
        // EXERCISE: Fill scratch view s_x.
        // ...
      }

      // EXERCISE: Add a barrier.
      // ...

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int j, double &innerUpdateN ) {
        double tempM = 0;

        Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( teamMember, M ), [&] ( const int i, double &innerUpdateM ) {
          // EXERCISE: Use the scratch s_x instead of x.
          innerUpdateM += A( e, j, i ) * x( e, i );
        }, tempM );

        innerUpdateN += y( e, j ) * tempM;
      }, tempN );

      Kokkos::single( Kokkos::PerTeam( teamMember ), [&] () {
        update += tempN;
      });
    }, result );

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d x %d is %lf\n", N, M, E, result );
    }

    const double solution = (double) N *(double) M *(double) E;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // The following is performed for each of E elements.
  //   Each matrix A row (each of length M) is read once.
  //   The x vector (of length M) is read N times.
  //   The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * E * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * E * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) E( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, E, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  }
  Kokkos::finalize();

  return 0;
}

void checkSizes( int &N, int &M, int &S, int &E, int &nrepeat ) {
  // If S is undefined and N or M is undefined, set S to 2^18 or the bigger of N and M.
  if ( S == -1 && ( N == -1 || M == -1 ) ) {
    S = pow( 2, 18 );
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

  // If E is undefined, set it to 2^10 = 1024.
  if ( E == -1 ) E = pow( 2, 10 );

  printf( "  Total size S = %d N = %d M = %d E = %d\n", S, N, M, E );

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
