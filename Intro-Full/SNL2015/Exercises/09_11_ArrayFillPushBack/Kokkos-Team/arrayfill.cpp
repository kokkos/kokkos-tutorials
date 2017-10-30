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

template< typename ExecutionSpace >
struct ArrayFillFunctor {

  typedef long value_type ;

  typedef typename Kokkos::TeamPolicy< ExecutionSpace >::member_type member_type ;

  typedef Kokkos::View< unsigned * , ExecutionSpace > array_type ;
  typedef Kokkos::View< unsigned ,   ExecutionSpace > count_type ;

  array_type      array ;
  count_type      count ;
  const unsigned  stride ;

  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & member , long & fail_count ) const
  {
    // Which index this member is evaluating
    const int i = member.league_rank() * member.team_size() + member.team_rank();

    // Is this index to be pushed into the array?
    const unsigned c = i % stride == 0 ? 1 : 0 ;

    // The team claims a range of locations in the array.
    // This member inputs how many locations it requires.
    const unsigned claim = member.team_scan( c , & count() );

    if ( c ) {
      if ( claim < array.dimension_0() ) {
        array[claim] = i ; 
      }
      else {
        ++fail_count ;
      }
    }
  }

  ArrayFillFunctor( const array_type & arg_array
                  , const count_type & arg_count
                  , const unsigned     arg_stride )
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

  // typedef Kokkos::Cuda ExecSpace ;
  typedef Kokkos::DefaultExecutionSpace ExecSpace ;

  // Allocate array to hold the number of push-backs

  const size_t length = 1 + ( number / stride );

  Kokkos::View< unsigned * , ExecSpace >  array("array",length);
  Kokkos::View< unsigned ,   ExecSpace >  count("count");

  Kokkos::View< unsigned ,   ExecSpace >::HostMirror  host_count =
   Kokkos::create_mirror_view( count );;
  
  // Timer
  double time = 0 ;
  struct timeval begin,end;

  for ( int k = 0 ; k < nrepeat ; ++k ) {
    // Re-initialize the count:
    host_count() = 0 ;

    Kokkos::deep_copy( count , host_count );

    gettimeofday(&begin,NULL);

    long fail_count = 0 ;

    const ArrayFillFunctor< ExecSpace > functor( array , count , stride );

    const unsigned team_size = Kokkos::TeamPolicy< ExecSpace >::team_size_max( functor );

    Kokkos::TeamPolicy< ExecSpace > policy( 1 + ( number / team_size ) , team_size );

    Kokkos::parallel_reduce( policy
                           , functor
                           , fail_count );

    gettimeofday(&end,NULL);

    time += 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);
  }

  // Print results

  printf("Number Density Time(s) TimePerIterations(s)\n");
  printf("%i %f %lf %e\n",number,(float)(1.0/(double)stride),time,time/nrepeat);

  Kokkos::finalize();

  return 0 ;
}

