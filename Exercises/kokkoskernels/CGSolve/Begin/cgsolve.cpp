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

// EXERCISE Goal: Implement conjugate gradient solver for square, symmetric, positive-definite sparse matrix using: 
//        - KokkosKernels BLAS functions
//        - KokkosKernels Sparse Matrix functions

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
// EXERCISE: Include header files for proper KokkosKernels BLAS and Sparse Matrix functions
// EXERCISE hint: KokkosBlas1_dot.hpp, KokkosBlas1_axpby.hpp, KokkosSparse_CrsMatrix.hpp, KokkosSparse_spmv.hpp

void checkSizes( int &N ) {
  // If N is undefined, set it to 2^10 = 1024.
  if ( N == -1 ) N = 1024;

  printf( "  Number of Rows N = %d, Number of Cols N = %d, Total nnz = %d\n", N, N, 2 + 3*(N - 2) + 2 );

  // Check sizes.
  if ( N < 0 ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }
}
template<typename crsMat_t>
void makeSparseMatrix (
      typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type & ptr,
      typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type   & ind,
      typename crsMat_t::values_type::non_const_type & val,
      typename crsMat_t::ordinal_type &numRows,
      typename crsMat_t::ordinal_type &numCols,
      typename crsMat_t::size_type &nnz,
      const int whichMatrix)
{


    typedef typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type ptr_type ;
    typedef typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type ind_type ;
    typedef typename crsMat_t::values_type::non_const_type val_type ;
    typedef typename crsMat_t::ordinal_type lno_t;
    typedef typename crsMat_t::size_type size_type;
    typedef typename crsMat_t::value_type scalar_t;

    using Kokkos::HostSpace;
    using Kokkos::MemoryUnmanaged;
    using Kokkos::View;

    if (whichMatrix == 0) {
      numCols = numRows;
      nnz     = 2 + 3*(numRows - 2) + 2;
      size_type* ptrRaw = new size_type[numRows + 1];
      lno_t*     indRaw = new lno_t[ nnz ];
      scalar_t* valRaw  = new scalar_t[ nnz ];
      scalar_t two  =  2.0;
      scalar_t mone = -1.0;

      // Add rows one-at-a-time
      for (int i = 0; i < (numRows + 1); i++) {
        if (i==0) {
           ptrRaw[0] = 0;
           indRaw[0] = 0;   indRaw[1] = 1;
           valRaw[0] = two; valRaw[1] = mone;
        }
        else if (i==numRows) {
           ptrRaw[numRows] = nnz;
        }
        else if (i==(numRows-1)) {
           ptrRaw[i] = 2 + 3*(i-1);
           indRaw[2 + 3*(i-1)] = i-1;  indRaw[2 + 3*(i-1) + 1] = i;
           valRaw[2 + 3*(i-1)] = mone; valRaw[2 + 3*(i-1) + 1] = two;
        }
        else {
           ptrRaw[i] = 2 + 3*(i-1);
           indRaw[2 + 3*(i-1)] = i-1;  indRaw[2 + 3*(i-1) + 1] = i;   indRaw[2 + 3*(i-1) + 2] = i+1;
           valRaw[2 + 3*(i-1)] = mone; valRaw[2 + 3*(i-1) + 1] = two; valRaw[2 + 3*(i-1) + 2] = mone;
        }
	  }

      // Create the output Views.
      ptr = ptr_type("ptr", numRows + 1);
      ind = ind_type("ind", nnz);
      val = val_type("val", nnz);

      // Wrap the above three arrays in unmanaged Views, so we can use deep_copy.
      typename ptr_type::HostMirror::const_type  ptrIn( ptrRaw , numRows+1 );
      typename ind_type::HostMirror::const_type  indIn( indRaw , nnz );
      typename val_type::HostMirror::const_type  valIn( valRaw , nnz );

      Kokkos::deep_copy (ptr, ptrIn);
      Kokkos::deep_copy (ind, indIn);
      Kokkos::deep_copy (val, valIn);

      delete[] ptrRaw;
      delete[] indRaw;
      delete[] valRaw;
    }
    else { // whichMatrix != 0
      std::ostringstream os;
      os << "Invalid whichMatrix value " << whichMatrix
         << ".  Valid value(s) include " << 0 << ".";
      throw std::invalid_argument (os.str ());
    }
}

template<typename crsMat_t>
crsMat_t  makeCrsMatrix (int numRows)
{
    typedef typename crsMat_t::StaticCrsGraphType graph_t;
    typedef typename graph_t::row_map_type::non_const_type lno_view_t;
    typedef typename graph_t::entries_type::non_const_type   lno_nnz_view_t;
    typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
    typedef typename crsMat_t::ordinal_type lno_t;
    typedef typename crsMat_t::size_type size_type;

    lno_view_t ptr;
    lno_nnz_view_t ind;
    scalar_view_t val;
    lno_t numCols;
    size_type nnz;

    const int whichMatrix = 0;
    makeSparseMatrix<crsMat_t> (ptr, ind, val, numRows, numCols, nnz, whichMatrix);
    return crsMat_t ("AA", numRows, numCols, nnz, val, ptr, ind);
}

int main( int argc, char* argv[] )
{
  srand(time(0));// Use current time as seed for random generator

  int N = -1;         // number of rows 2^10
  
  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( strcmp( argv[ i ], "-N" ) == 0 ) {
      N = atoi( argv[ ++i ] );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  CGSolve Options:\n" );
      printf( "  -N <int>:      determines number of rows (columns) (default: 2^10 = 1024)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N );
  
  Kokkos::initialize( argc, argv );
  {
    //typedef Kokkos::DefaultExecutionSpace::array_layout  Layout;
    //typedef Kokkos::LayoutLeft   Layout;
    typedef Kokkos::LayoutRight  Layout;

    typedef Kokkos::View<double*, Layout>  ViewVectorType;

    // Timer products.
    struct timeval begin, end, c0, c1;

    //
    double one   = 1.0;
    double zero  = 0.0;
    double tolerance = 0.0000000001;

    // Allocate vectors and Matrix A on device.
    typedef KokkosSparse::CrsMatrix<double, int, Kokkos::DefaultExecutionSpace, void, int> crs_matrix_type;
    crs_matrix_type A = makeCrsMatrix<crs_matrix_type> (N);

    ViewVectorType b    ( "b",    N );
    ViewVectorType x    ( "x",    N );
    ViewVectorType xx   ( "xx",   N );
	
    ViewVectorType p    ( "p",    N );
    ViewVectorType Ap   ( "Ap",   N );
    ViewVectorType r    ( "r",    N );

    // Create host mirrors of device views.
    ViewVectorType::HostMirror h_xx = Kokkos::create_mirror_view( xx );

    // Initialize xx vector on host.
    for(int i=0; i<xx.span(); i++) h_xx.data()[i] = (double)rand() / RAND_MAX;

    Kokkos::deep_copy( xx, h_xx );

    // EXERCISE: Generate RHS vector b by multilying A with the reference solution xx 
    /* b = A*xx */
    // EXERCISE hint: KokkosSparse::spmv
	
    // Deep copy b to r
    Kokkos::deep_copy( r, b );

    Kokkos::fence();

    // Start CGSolve
    gettimeofday( &begin, NULL );
    gettimeofday( &c0, NULL );
    // EXERCISE:
    /* r = b - A*x */
    // EXERCISE hint: KokkosSparse::spmv, KokkosBlas::axpy
	
    // EXERCISE:
    /* r_old_dot = <r,r> */
    // EXERCISE hint: KokkosBlas::dot

    double norm_res  = std::sqrt( r_old_dot );

    gettimeofday( &c1, NULL );
    double tt1 = 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
                 1.0e-6 * ( c1.tv_usec - c0.tv_usec );

    gettimeofday( &c0, NULL );
    /* p  = r */
    Kokkos::deep_copy( p, r ); Kokkos::fence();
    gettimeofday( &c1, NULL );
    double tt2 = 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
                 1.0e-6 * ( c1.tv_usec - c0.tv_usec );

    int k = 0 ;

    double tt3 = 0.0;
    double tt4 = 0.0;

    while ( tolerance < norm_res && k < N ) {
        gettimeofday( &c0, NULL );
        // EXERCISE: 
        /* alpha = (r'*r)/(p'*A*p) */
        // EXERCISE hint: KokkosSparse::spmv, KokkosBlas::dot
        gettimeofday( &c1, NULL );
        tt3 += 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
               1.0e-6 * ( c1.tv_usec - c0.tv_usec );

        gettimeofday( &c0, NULL );
        // EXERCISE:
        /* x = x + alpha*p */
        /* r = r - alpha*A*p */
        // EXERCISE hint: KokkosBlas::axpy

        // EXERCISE: beta = (r'*r)/(r_old'*r_old)
        /* r_dot = <r,r> */
        /* beta = r_dot/r_old_dot */
        // EXERCISE hint: KokkosBlas::dot

        // EXERCISE:
        /* p = r + beta * p */
        // EXERCISE hint: KokkosBlas::axpby

        norm_res = std::sqrt( r_old_dot = r_dot );

        k++ ;

        Kokkos::fence();
        gettimeofday( &c1, NULL );
        tt4 += 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
               1.0e-6 * ( c1.tv_usec - c0.tv_usec );
    }

    Kokkos::fence();

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 *    ( end.tv_sec  - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );

    // Check result
    KokkosBlas::axpby(one, x, -one, xx);
    double final_norm_res  = std::sqrt( KokkosBlas::dot(xx, xx) );

    // Print results (problem size, time, number of iterations and final norm residual).
    printf( "    Results: N( %d ), time( %g s ), iterations( %d ), final norm_res(%.20lf), time part1( %g s ), time part2( %g s ), time part3( %g s ), time part4( %g s )\n",
            N, time, k, final_norm_res, tt1, tt2, tt3, tt4 );
  }

  Kokkos::finalize();

  return 0;
}
