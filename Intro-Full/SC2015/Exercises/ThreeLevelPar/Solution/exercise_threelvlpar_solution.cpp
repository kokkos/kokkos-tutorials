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

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

 #include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
  int N = 4096 ;       // number of rows 2^12
  int M = 1024 ;       // number of columns 2^10
  int nrepeat = 100 ;    // number of repeats of the test
  int S = N * M ;      // total size 2^22

  // Read command line arguments
  for(int i=0; i<argc; i++) {
           if( (strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0)) {
      N = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-M") == 0) || (strcmp(argv[i], "-Columns") == 0)) {
      M = pow( 2, atof(argv[++i]) );
      //M = atof(argv[++i]);
    } else if( (strcmp(argv[i], "-S") == 0) || (strcmp(argv[i], "-Size") == 0)) {
      S = pow( 2, atof(argv[++i]) );
      //S = atof(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("  y^T*A*x Options:\n");
      printf("  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n");
      printf("  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n");
      printf("  -nrepeat <int>:        number of repetitions (default: 100)\n");
      printf("  -help (-h):            print this message\n\n");
    }
  }

  std::cout << " Running for S = "<<S<<"  M = "<<M<<std::endl;

  //Check Sizes
  if ( (S < 0) || (M < 0) || (N < 0 ) || (nrepeat < 0 ) ) {
    printf("  Sizes must be greater than 0\n");
    exit (1);
  }

  if ( S < M ) {
    printf(" Problem size S must be larger than M\n");
    exit (2);
  }

  N = (int) (S / M);
  if ( (N * M) != S ) {
    printf("  M is not a factor of S, adjusting value...\n");
  }

  while ( ( (N * M) != S ) && ( (N * M) <= S) ) {
    M += 1;
    N = (int) (S / M);
  }

  if ( N < 128 ) {
    printf("  S and M combination must be such that N >= 128 for third level par. \n");
    exit (1);
  }

  if ( (N * M) != S ) {
    printf(" Cannot adjust M properly, choose different S, M combination \n");
    exit (3);
  }


  Kokkos::initialize(argc,argv);

  // typedef Kokkos::Serial   Space ;
  // typedef Kokkos::Threads  Space ;
  // typedef Kokkos::OpenMP   Space ;
  typedef Kokkos::Cuda     Space ;

  // typedef Kokkos::LayoutLeft   Layout ;
  typedef Kokkos::LayoutRight  Layout ;

  typedef Kokkos::View<double*, Space>   ViewVector;
  typedef Kokkos::View<double**, Layout, Space>   ViewMatrix;
  ViewVector y("y", N);
  ViewVector x("x", M);
  ViewMatrix A("A", N, M);

  typedef Kokkos::RangePolicy<Space> range_policy ;

  // Initialize y vector
  Kokkos::parallel_for( range_policy( 0 , N ), KOKKOS_LAMBDA( const int i ) {
    y( i ) = 1; 
  } );

  // Initialize x vector
  Kokkos::parallel_for( range_policy( 0 , M ), KOKKOS_LAMBDA( const int i ) {
    x( i ) = 1;
  } );

  typedef Kokkos::TeamPolicy<Space>               team_policy ;
  typedef Kokkos::TeamPolicy<Space>::member_type  member_type ;

  // Initialize A matrix, note 2D indexing computation
  Kokkos::parallel_for( team_policy( N, Kokkos::AUTO ), KOKKOS_LAMBDA( const member_type& teamMember ) {
    const int j = teamMember.league_rank();
    Kokkos::parallel_for( Kokkos::TeamThreadRange( teamMember, M ), [&] ( const int i ) { 
      A( j , i ) = 1; 
    } );
  } );

  // Timer products
  struct timeval begin,end;

  gettimeofday(&begin,NULL);

  for ( int repeat = 0; repeat < nrepeat; repeat++) {

  //Application: <y,Ax> = y^T*A*x
  // Three level parallelism kernel to force caching of vector x 
  Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess> > x_r = x;
  Kokkos::View<double*,Space> updates("updates", N);
  double result = 0.0;

  Kokkos::parallel_for( team_policy( N/128 , 32 , 32 ), KOKKOS_LAMBDA ( const member_type& teamMember) {
      const int row_start = teamMember.league_rank()*128;
      Kokkos::parallel_for( Kokkos::TeamThreadRange(teamMember, row_start, row_start + 128), [&] (const int i) {
        double sum_i = 0.0;
        Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(teamMember, M), [&] (const int j, double &innerUpdate ) {
          innerUpdate += A( i , j ) * x_r( j );
        }, sum_i);
        Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
          updates(i) = y(i) * sum_i;
        });
      });
    });
    Kokkos::parallel_reduce(N,KOKKOS_LAMBDA (const int& i, double& update) {
      update+=updates(i);
    },result);


   //Output final result
   if ( repeat == ( nrepeat - 1) )
     printf("  Computed result for %d x %d is %lf\n", N, M, result);
   const double solution= (double)N * (double)M ; 

    if ( result != solution ) {
      printf("  Error: result( %lf ) != solution( %lf )\n",result,solution);
    }
  }

  gettimeofday(&end,NULL);

  // Calculate time
  double time = 1.0*(end.tv_sec-begin.tv_sec) +
                1.0e-6*(end.tv_usec-begin.tv_usec);

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times, or once if caching...
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double(sizeof(double) * ( 2 * M * N + N ));
  double Gbytes = 1.0e-9 * double(sizeof(double) * ( M + M * N + N )) * nrepeat;

  // Print results (problem size, time and bandwidth in GB/s)
  printf("  M( %d ) N( %d ) nrepeat ( %d ) problem( %g GB ) time( %g s ) bandwidth( %g GB/s )\n",
         M , N, nrepeat, Gbytes, time, Gbytes / time );

  Kokkos::finalize();

  return 0 ;
}

