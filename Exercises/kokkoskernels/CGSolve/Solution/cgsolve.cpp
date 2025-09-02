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
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

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
      typename ptr_type::host_mirror_type::const_type  ptrIn( ptrRaw , numRows+1 );
      typename ind_type::host_mirror_type::const_type  indIn( indRaw , nnz );
      typename val_type::host_mirror_type::const_type  valIn( valRaw , nnz );

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
    // Define scalar types based on availability of kokkos half precision support
#if defined(KOKKOS_HALF_T_IS_FLOAT)
    using ScalarType = Kokkos::Experimental::half_t;
    using AccumulatorType = float;
#else
    using ScalarType = double;
    using AccumulatorType = ScalarType;
#endif // KOKKOS_HALF_T_IS_FLOAT

    // Define view template argument types
#if defined(KOKKOS_ENABLE_CUDA)
    using DeviceType = Kokkos::Cuda;
#else
    using DeviceType = Kokkos::HostSpace;
#endif // defined(KOKKOS_ENABLE_CUDA)
    using ExecutionSpace = DeviceType::execution_space;
    //using Layout = ExecutionSpace::array_layout;
    //using Layout = Kokkos::LayoutLeft;
    using Layout = Kokkos::LayoutRight;

    // Define view types
    using ViewVectorType = Kokkos::View<ScalarType*, Layout, DeviceType>;
    using AccumulatorVectorType = Kokkos::View<AccumulatorType, Layout, DeviceType>;

    // Declare timer products.
    struct timeval begin, end, c0, c1;

    // Defined scalars used in loop
    ScalarType one   = 1.0;
    ScalarType zero  = 0.0;
    ScalarType tolerance = 0.0001; // Smallest positive half_t is 0
    // .00006103515625.

    // Declare denominator for random number generation
    ScalarType rand_max;

    // Initialize rand_max based on ScalarType in use
#if defined(KOKKOS_HALF_T_IS_FLOAT)
    if (std::is_same<ScalarType, Kokkos::Experimental::half_t>::value)
      rand_max = static_cast<ScalarType>(0x7BFF); // Largest positive half_t is 0x7BFF.
    else
      rand_max = RAND_MAX;
#else
    rand_max = RAND_MAX;
#endif // KOKKOS_HALF_T_IS_FLOAT

    // Allocate vectors and Matrix A on device.
    typedef KokkosSparse::CrsMatrix<ScalarType, int, ExecutionSpace, void, int> crs_matrix_type;
    crs_matrix_type A = makeCrsMatrix<crs_matrix_type> (N);

    ViewVectorType b    ( "b",    N );
    ViewVectorType x    ( "x",    N );
    ViewVectorType xx   ( "xx",   N );

    ViewVectorType p    ( "p",    N );
    ViewVectorType Ap   ( "Ap",   N );
    ViewVectorType r    ( "r",    N );

    // Allocate 0-D vector on device for dot product accumulation
    AccumulatorVectorType dot_ret ("dot_ret");

    // Create host mirrors of device views.
    auto h_xx = Kokkos::create_mirror_view( xx );
    auto h_dot_ret = Kokkos::create_mirror_view( dot_ret );

    // Initialize h_xx with random numbers in [0, 1]
    for(int i=0; i<xx.span(); i++) {
#if defined(KOKKOS_HALF_T_IS_FLOAT)
      // Produce a random half_t between 0 and 1. Mask 32 bit random number off to last 14 bits.
      if (std::is_same<ScalarType, Kokkos::Experimental::half_t>::value)
	h_xx.data()[i] = static_cast<ScalarType>(rand() & 0x3FFF) / rand_max;
      else
	h_xx.data()[i] = static_cast<ScalarType>(rand()) / rand_max;
#else
      h_xx.data()[i] = static_cast<ScalarType>(rand()) / rand_max;
#endif // KOKKOS_HALF_T_IS_FLOAT
      // Check for random numbers outside of [0, 1]
      if (h_xx.data()[i] > 1 || h_xx.data()[i] < 0) {
	printf("ERROR: h_xx.data()[%d] = %g not it [0, 1].\n", i,
	       static_cast<double>(h_xx.data()[i]));
	exit(-1);
      }
    }

    // Copy h_xx from host to device
    Kokkos::deep_copy( xx, h_xx );

    // EXERCISE: Generate RHS vector b by multilying A with the reference solution xx
    /* b = A*xx */
    // EXERCISE hint: KokkosSparse::spmv
    // b = 0*b + 1*A*xx
    KokkosSparse::spmv( "N" , one , A , xx , zero , b);

    // Deep copy b to r.
    Kokkos::deep_copy( r, b );

    // Start CGSolve
    gettimeofday( &begin, NULL );
    gettimeofday( &c0, NULL );
    // EXERCISE:
    /* r = b - A*x */
    // EXERCISE hint: KokkosSparse::spmv, KokkosBlas::axpy
    // Ap = 0*Ap + 1*A*x
    KokkosSparse::spmv( "N" , one , A , x , zero , Ap);

    // r = b - Ap
    KokkosBlas::axpy(-1.0, Ap, r);

    // EXERCISE:
    /* dot_ret = <r,r> */
    // EXERCISE hint: KokkosBlas::dot
    KokkosBlas::dot(dot_ret, r,r);

    // Ensure that dot completes on device before copying result to host
    ExecutionSpace().fence();
    Kokkos::deep_copy(h_dot_ret, dot_ret);

    AccumulatorType r_old_dot = h_dot_ret.data()[0];
    AccumulatorType norm_res  = std::sqrt( static_cast<double>(r_old_dot) );

    gettimeofday( &c1, NULL );
    double tt1 = 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
                 1.0e-6 * ( c1.tv_usec - c0.tv_usec );

    gettimeofday( &c0, NULL );
    /* p  = r */
    Kokkos::deep_copy( p, r );

    gettimeofday( &c1, NULL );
    double tt2 = 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
                 1.0e-6 * ( c1.tv_usec - c0.tv_usec );

    int k = 0 ;

    double tt3 = 0.0;
    double tt4 = 0.0;

    printf("tolerance(%g), norm_res(%g), k(%d), N(%d)\n", static_cast<double>(tolerance), static_cast<double>(norm_res), k, N);

    while ( tolerance < norm_res && k < N ) {
        gettimeofday( &c0, NULL );
        // EXERCISE:
        /* alpha = (r'*r)/(p'*A*p) */
        // EXERCISE hint: KokkosSparse::spmv, KokkosBlas::dot
        // Ap = A * p
        KokkosSparse::spmv( "N" , one , A , p , zero , Ap);

        // pAp_dot = p'*A*p
	KokkosBlas::dot(dot_ret, p, Ap);

	// Ensure that dot completes on device before copying result to host
	ExecutionSpace().fence();
	Kokkos::deep_copy(h_dot_ret, dot_ret);
	AccumulatorType pAp_dot = h_dot_ret.data()[0];

        // alpha = r_old_dot/pAp_dot
        ScalarType alpha   = r_old_dot / pAp_dot ;
        gettimeofday( &c1, NULL );
        tt3 += 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
               1.0e-6 * ( c1.tv_usec - c0.tv_usec );

        gettimeofday( &c0, NULL );
        // EXERCISE:
        /* x = x + alpha*p */
        /* r = r - alpha*A*p */
        // EXERCISE hint: KokkosBlas::axpy
        KokkosBlas::axpy(alpha, p, x);

        KokkosBlas::axpy(-alpha, Ap, r);

        // EXERCISE: beta = (r'*r)/(r_old'*r_old)
        /* r_dot = <r,r> */
        /* beta = r_dot/r_old_dot */
        // EXERCISE hint: KokkosBlas::dot
        // r_dot = r'*r
	KokkosBlas::dot(dot_ret,r,r);

	// Ensure that dot completes on device before copying result to host
	ExecutionSpace().fence();
	Kokkos::deep_copy(h_dot_ret, dot_ret);

	AccumulatorType r_dot = h_dot_ret.data()[0];
        ScalarType beta  = r_dot / r_old_dot;

        // EXERCISE:
        /* p = r + beta * p */
        // EXERCISE hint: KokkosBlas::axpby
        KokkosBlas::axpby(one, r, beta, p);

        norm_res = std::sqrt( static_cast<double>(r_old_dot = r_dot) );

        k++;

	// Ensure device kernels are done before taking time stamp
	ExecutionSpace().fence();

        gettimeofday( &c1, NULL );
        tt4 += 1.0 *    ( c1.tv_sec  - c0.tv_sec ) +
               1.0e-6 * ( c1.tv_usec - c0.tv_usec );
    }

    // Ensure device kernels are done before taking final time stamp
    ExecutionSpace().fence();

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 *    ( end.tv_sec  - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );

    // Check result
    KokkosBlas::axpby(one, x, -one, xx);
    KokkosBlas::dot(dot_ret, xx, xx);

    // Ensure that dot completes on device before copying result to host
    ExecutionSpace().fence();

    Kokkos::deep_copy(h_dot_ret, dot_ret);
    AccumulatorType final_norm_res  = std::sqrt( static_cast<double>(h_dot_ret.data()[0]) );

    // Print results (problem size, time, number of iterations and final norm residual).
    printf( "    Results: N( %d ), time( %g s ), iterations( %d ), final norm_res(%g), norm_res(%g), tolerance(%g), time part1( %g s ), time part2( %g s ), time part3( %g s ), time part4( %g s )\n",
            N, time, k, static_cast<double>(final_norm_res), static_cast<double>(norm_res), static_cast<double>(tolerance), tt1, tt2, tt3, tt4 );
  }

  Kokkos::finalize();

  return 0;
}
