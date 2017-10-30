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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#include <cstdio>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

struct ArrayFillFunctor {

  typedef long value_type ;

  typedef Kokkos::View< unsigned * > array_type ;
  typedef Kokkos::View< unsigned > count_type ;

  array_type      array ;
  count_type      count ;
  const unsigned  stride ;

  KOKKOS_INLINE_FUNCTION
  void operator()( const unsigned i , long & failed_insert_count ) const
  {
    if ( i % stride == 0 ) {
      const unsigned claim = Kokkos::atomic_fetch_add( & count() , 1 );
      if ( claim < array.dimension_0() ) {
        array[claim] = i ; 
      }
      else {
        ++failed_insert_count ;
      }
    }
  }

  ArrayFillFunctor( const array_type & arg_array
                  , const count_type & arg_count
                  , const unsigned    arg_stride )
    : array( arg_array )
    , count( arg_count )
    , stride( arg_stride )
    {}
};

int main( int argc , char * argv[] )
{
  // Parameters
  int number  = 10000000 ;
  int stride  = 100 ;
  int nrepeat = 10 ;
  
  // Read command line arguments
  for(int i=0; i<argc; i++) {
    if( strcmp(argv[i], "-n") == 0) {
      number = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-s") == 0) {
      stride = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("ArrayFill Options:\n");
      printf("  -n <int>:         number parallel iterations (default: 10000000)\n");
      printf("  -s <int>:         stride of push-back (default: 100)\n");
      printf("  -nrepeat <int>:   number of repets (default: 10)\n");
      printf("  -help (-h):       print this message\n");
    }
  }

  Kokkos::initialize( argc , argv );

  // Allocate array to hold the number of push-backs

  size_t length = 100;

  Kokkos::View< unsigned * >  array("array",length);
  Kokkos::View< unsigned >  count("count");

  Kokkos::View< unsigned >::HostMirror  host_count =
   Kokkos::create_mirror_view( count );;
  
  // Timer
  double time = 0 ;
  struct timeval begin,end;

  gettimeofday(&begin,NULL);

  for ( int k = 0 ; k < nrepeat ; ++k ) {
    // Re-initialize the count:
    host_count() = 0 ;

    Kokkos::deep_copy( count , host_count );


    long failed_insert_count = 0 ;

    // Fill the array and count failed to insert
    Kokkos::parallel_reduce( Kokkos::RangePolicy<>( 0 , number )
                           , ArrayFillFunctor( array , count , stride )
                           , failed_insert_count );


    // If you ran out of space, reallocate array and refill
    if ( failed_insert_count ) {
      array = Kokkos::View< unsigned * > ("array", length + failed_insert_count );
      length += failed_insert_count;

      // Re-initialize the count:
      host_count() = 0 ;

      Kokkos::deep_copy( count , host_count );

      Kokkos::parallel_reduce( Kokkos::RangePolicy<>( 0 , number )
                             , ArrayFillFunctor( array , count , stride )
                             , failed_insert_count );
    }
  }

  gettimeofday(&end,NULL);

  time += 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);

  // Print results

  printf("Number Density Time(s) TimePerIterations(s)\n");
  printf("%i %f %lf %e\n",number,(float)(1.0/(double)stride),time,time/nrepeat);

  Kokkos::finalize();

  return 0 ;
}

