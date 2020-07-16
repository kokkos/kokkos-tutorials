/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// EXERCISE Goal: Implement a preconditioned conjugate gradient solver for square, symmetric, 
//                positive-definite sparse matrix using: 
//                - KokkosKernels BLAS functions
//                - KokkosKernels Sparse Matrix functions:
//                + KokkosKernels Sparse ILU(k)
//                + KokkosKernels Sparse tri-solve

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
// EXERCISE: Include header files for proper KokkosKernels BLAS and Sparse Matrix functions
// EXERCISE hint: KokkosBlas1_dot.hpp, KokkosBlas1_axpby.hpp, KokkosSparse_CrsMatrix.hpp, KokkosSparse_spmv.hpp, KokkosSparse_spiluk.hpp, KokkosSparse_sptrsv.hpp

#include <KokkosKernels_IOUtils.hpp>
#include <KokkosKernels_Test_Structured_Matrix.hpp>

#define EXPAND_FACT 6 // a factor used in expected sizes of L and U

enum {DEFAULT, LVLSCHED_RP, LVLSCHED_TP1};

int main( int argc, char* argv[] )
{
  srand(12345); // Fix a seed for random generator

  using scalar_t  = double;
  using lno_t     = int;
  using size_type = int;

  int nx = 10;                   // grid points in 'x' direction
  int ny = 10;                   // grid points in 'y' direction
  int algo_spiluk = LVLSCHED_RP; // SPILUK kernel implementation
  int algo_sptrsv = LVLSCHED_RP; // SPTRSV kernel implementation
  int k = 0;                     // fill level
  int team_size = -1;            // team size
  int prec = 0;                  // preconditioning flag 

  // Read command line arguments
  for ( int i = 0; i < argc; i++ ) {
    if ( strcmp( argv[ i ], "-nx" ) == 0 ) {
      nx = atoi( argv[ ++i ] );
      printf( "  User's grid points in x direction is %d\n", nx );
    }
    else if ( strcmp( argv[ i ], "-ny" ) == 0 ) {
      ny = atoi( argv[ ++i ] );
      printf( "  User's grid points in y direction is %d\n", ny );
    }
    else if ( strcmp( argv[ i ], "-k" ) == 0 ) {
      k = atoi( argv[ ++i ] );
      printf( "  User's fill level is %d\n", k );
    }
    else if ( strcmp( argv[ i ], "-algospiluk" ) == 0 ) {
      i++;
      if ( strcmp( argv[ i ], "lvlrp" ) == 0 ) {
        algo_spiluk = LVLSCHED_RP;
      }
      if ( strcmp( argv[ i ], "lvltp1" ) == 0 ) {
        algo_spiluk = LVLSCHED_TP1;
      }
    }
    else if ( strcmp( argv[ i ], "-algosptrsv" ) == 0 ) {
      i++;
      if ( strcmp( argv[ i ], "lvlrp" ) == 0 ) {
        algo_sptrsv = LVLSCHED_RP;
      }
      if ( strcmp( argv[ i ], "lvltp1" ) == 0 ) {
        algo_sptrsv = LVLSCHED_TP1;
      }
    }
    else if ( strcmp( argv[ i ], "-ts" ) == 0 ) {
      team_size = atoi( argv[ ++i ] );
      printf( "  User's team_size is %d\n", team_size );
    }
    else if ( strcmp( argv[ i ], "-prec" ) == 0 ) {
      prec = atoi( argv[ ++i ] );
      printf( "  User's preconditioning flag is %d\n", prec );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  CGSolve with ILU(k) preconditioner options: (simple Laplacian matrix on a cartesian grid where nrows = nx * ny)\n" );
      printf( "  -nx <int>           : grid points in x direction (default: 10)\n" );
      printf( "  -ny <int>           : grid points in y direction (default: 10)\n" );
      printf( "  -k <int>            : Fill level in SPILUK (default: 0)\n" );
      printf( "  -algospiluk [OPTION]: SPILUK kernel implementation (default: lvlrp)\n" );
      printf( "  -algosptrsv [OPTION]: SPTRSV kernel implementation (default: lvlrp)\n" );
      printf( "                        [OPTIONS]: lvlrp, lvltp1\n");
      printf( "  -ts <int>           : team size only when lvltp1 is used (default: -1)\n");
      printf( "  -prec <int>         : whether having preconditioner or not (default: 0)\n");
      printf( "  -help (-h)          : Print this message\n\n" );
      exit( 1 );
    }
  }
  
  Kokkos::initialize( argc, argv );
  {
    using Layout          = Kokkos::LayoutLeft;
    using ViewVectorType  = Kokkos::View<scalar_t*, Layout>;
    using execution_space = typename ViewVectorType::device_type::execution_space;
    using memory_space    = typename ViewVectorType::device_type::memory_space;
    using crsmat_t        = KokkosSparse::CrsMatrix<scalar_t, lno_t, Kokkos::DefaultExecutionSpace, void, size_type>;
    using graph_t         = typename crsmat_t::StaticCrsGraphType;
    using lno_view_t      = typename graph_t::row_map_type::non_const_type; //row map view type
    using lno_nnz_view_t  = typename graph_t::entries_type::non_const_type; //entries view type
    using scalar_view_t   = typename crsmat_t::values_type::non_const_type; //values view type
    using KernelHandle    = KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
                                                                              execution_space, memory_space, memory_space>;

    std::cout << "CGSolve execution_space: " << typeid(execution_space).name() << ", memory_space: " << typeid(memory_space).name() << std::endl;

    // Timer products.
    struct timeval begin, end;

    //
    scalar_t tolerance = 0.0000000001;
    scalar_t one  = scalar_t(1.0);
    scalar_t zero = scalar_t(0.0);
    scalar_t mone = scalar_t(-1.0);

    // Generate a simple Laplacian matrix on a cartesian grid.
    // The mat_structure view is used to generate a matrix using
    // finite difference (FD) or finite element (FE) discretization
    // on a cartesian grid.
    // Each row corresponds to an axis (x, y and z)
    // In each row the first entry is the number of grid point in
    // that direction, the second and third entries are used to apply
    // BCs in that direction, BC=0 means Neumann BC is applied,
    // BC=1 means Dirichlet BC is applied by zeroing out the row and putting
    // one on the diagonal.
    Kokkos::View<lno_t*[3], Kokkos::HostSpace> mat_structure("Matrix Structure", 2);
    mat_structure(0, 0) = nx;  // Request nx grid point in 'x' direction
    mat_structure(0, 1) = 0;   // Add BC to the left
    mat_structure(0, 2) = 0;   // Add BC to the right
    mat_structure(1, 0) = ny;  // Request ny grid point in 'y' direction
    mat_structure(1, 1) = 0;   // Add BC to the bottom
    mat_structure(1, 2) = 0;   // Add BC to the top

    crsmat_t A = Test::generate_structured_matrix2D<crsmat_t>("FD", mat_structure);

    graph_t  graph    = A.graph; // in_graph
    const size_type N = graph.numRows();
    typename KernelHandle::const_nnz_lno_t fill_lev = lno_t(k) ;
    const size_type nnzA = A.graph.entries.extent(0);
    std::cout << "Matrix size: " << N << " x " << N << ", nnz = " << nnzA << std::endl;

    // Create SPILUK handle and SPTRSV handles (for L and U)
    KernelHandle kh_spiluk, kh_sptrsv_L, kh_sptrsv_U;

    std::cout << "Create SPILUK handle ..." << std::endl;
    switch(algo_spiluk) {
      case LVLSCHED_RP: //Using range policy (KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP)
        kh_spiluk.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1));
        std::cout << "Kernel implementation type: "; kh_spiluk.get_spiluk_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1: //Using team policy (KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1)
        kh_spiluk.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1));
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh_spiluk.get_spiluk_handle()->set_team_size(team_size);
        std::cout << "Kernel implementation type: "; kh_spiluk.get_spiluk_handle()->print_algorithm();
        break;
      default: //Using range policy
        kh_spiluk.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1));
        std::cout << "Kernel implementation type: "; kh_spiluk.get_spiluk_handle()->print_algorithm();
    }

    // EXERCISE: Create a SPTRSV handle
    // EXERCISE hint: input arguments include implementation type (i.e. KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP or KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1), number of matrix rows, lower or upper matrix flag (boolean)
    std::cout << "Create SPTRSV handle for L ..." << std::endl;
    bool is_lower_tri = true;
    switch(algo_sptrsv) {
      case LVLSCHED_RP:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        
        kh_sptrsv_L.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh_sptrsv_L.get_sptrsv_handle()->set_team_size(team_size);
        kh_sptrsv_L.get_sptrsv_handle()->print_algorithm();
        break;
      default:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        
        kh_sptrsv_L.get_sptrsv_handle()->print_algorithm();
    }

    std::cout << "Create SPTRSV handle for U ..." << std::endl;
    is_lower_tri = false;
    switch(algo_sptrsv) {
      case LVLSCHED_RP:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        
        kh_sptrsv_U.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh_sptrsv_U.get_sptrsv_handle()->set_team_size(team_size);
        kh_sptrsv_U.get_sptrsv_handle()->print_algorithm();
        break;
      default:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        
        kh_sptrsv_U.get_sptrsv_handle()->print_algorithm();
    }
	
    auto spiluk_handle  = kh_spiluk.get_spiluk_handle();
    auto sptrsvL_handle = kh_sptrsv_L.get_spiluk_handle();
    auto sptrsvU_handle = kh_sptrsv_U.get_spiluk_handle();

    // Allocate row map, entries, values views (1-D) for L and U
    lno_view_t     L_row_map("L_row_map", N + 1);
    lno_nnz_view_t L_entries("L_entries", spiluk_handle->get_nnzL());
    scalar_view_t  L_values ("L_values",  spiluk_handle->get_nnzL());
    lno_view_t     U_row_map("U_row_map", N + 1);
    lno_nnz_view_t U_entries("U_entries", spiluk_handle->get_nnzU());
    scalar_view_t  U_values ("U_values",  spiluk_handle->get_nnzU());	

    // ILU(k) Symbolic phase
    std::cout << "Run ILU(k) symbolic phase ..." << std::endl;
    gettimeofday( &begin, NULL );
    KokkosSparse::Experimental::spiluk_symbolic( &kh_spiluk, fill_lev, 
                                                 A.graph.row_map, A.graph.entries, 
                                                 L_row_map, L_entries, U_row_map, U_entries );
    Kokkos::fence();
    gettimeofday( &end, NULL );
    std::cout << "ILU(" << fill_lev << ") Symbolic Time: " << 1.0 * ( end.tv_sec  - begin.tv_sec  ) +
                                                           1.0e-6 * ( end.tv_usec - begin.tv_usec ) << " seconds" << std::endl;

    // Resize L and U to their actual sizes
    Kokkos::resize(L_entries, spiluk_handle->get_nnzL());
    Kokkos::resize(L_values,  spiluk_handle->get_nnzL());
    Kokkos::resize(U_entries, spiluk_handle->get_nnzU());
    Kokkos::resize(U_values,  spiluk_handle->get_nnzU());

    std::cout << "L_row_map size = " << L_row_map.extent(0) << std::endl;
    std::cout << "L_entries size = " << L_entries.extent(0) << std::endl;
    std::cout << "L_values size  = " << L_values.extent(0)  << std::endl;
    std::cout << "U_row_map size = " << U_row_map.extent(0) << std::endl;
    std::cout << "U_entries size = " << U_entries.extent(0) << std::endl;
    std::cout << "U_values size  = " << U_values.extent(0)  << std::endl;
    std::cout << "ILU(k) fill_level: "   << fill_lev << std::endl;
    std::cout << "ILU(k) fill-factor: "  << (spiluk_handle->get_nnzL() + spiluk_handle->get_nnzU() - N)/(double)nnzA << std::endl;
    std::cout << "num levels: "          << spiluk_handle->get_num_levels() << std::endl;
    std::cout << "max num rows levels: " << spiluk_handle->get_level_maxrows() << std::endl << std::endl;

    // ILU(k) Numeric phase
    std::cout << "Run ILU(k) numeric phase ..." << std::endl;
    gettimeofday( &begin, NULL );
    KokkosSparse::Experimental::spiluk_numeric( &kh_spiluk, fill_lev, 
                                                 A.graph.row_map, A.graph.entries, A.values, 
                                                 L_row_map, L_entries, L_values, U_row_map, U_entries, U_values );
    Kokkos::fence();
    gettimeofday( &end, NULL );
    std::cout << "ILU(" << fill_lev << ") Numeric Time: " << 1.0 * ( end.tv_sec  - begin.tv_sec  ) +
                                                          1.0e-6 * ( end.tv_usec - begin.tv_usec ) << " seconds" << std::endl;
														  
    // Tri-solve Symbolic phase
    // EXERCISE: Run symbolic phase
    // EXERCISE hint: KokkosSparse::Experimental::sptrsv_symbolic(...)


    // Allocate vectors needed for CGSolve
    ViewVectorType b    ( "b",     N );
    ViewVectorType x    ( "x",     N );
    ViewVectorType xx   ( "xx",    N );

    ViewVectorType p    ( "p",     N );
    ViewVectorType Ap   ( "Ap",    N );
    ViewVectorType r    ( "r",     N );
    ViewVectorType z    ( "z",     N );
    ViewVectorType Linvr( "Linvr", N );

    ViewVectorType::HostMirror h_xx = Kokkos::create_mirror_view( xx );

    // Initialize xx vector on host
    for(int i=0; i<xx.span(); i++) h_xx.data()[i] = (double)rand() / RAND_MAX;

    Kokkos::deep_copy( xx, h_xx );

    // Generate RHS vector b by multilying A with the reference solution xx 
    /* b = A*xx */
    KokkosSparse::spmv( "N" , one , A , xx , zero , b);
	
    // Deep copy b to r
    Kokkos::deep_copy( r, b ); Kokkos::fence();

    // Start CGSolve
    gettimeofday( &begin, NULL );

    /* r = b - A*x */
    // Ap = A * x
    KokkosSparse::spmv( "N" , one , A , x , zero , Ap); Kokkos::fence();
    // r = b - Ap
    KokkosBlas::axpy(mone, Ap, r); Kokkos::fence();

    if (prec == 1) {
      // EXERCISE: z = inv(M)*r or solve M*z = r for z or solve (L*U)*z = r for z
      // EXERCISE hint: solve L*Linvr = r     for Linvr, then
      //                solve U*z     = Linvr for z
      //                KokkosSparse::Experimental::sptrsv_solve(...)	  

    }
    else {
      Kokkos::deep_copy( z, r );
    }

    /* r_old_dot = <r,z> */
    double r_old_dot = KokkosBlas::dot(r,z);

    double norm_res  = std::sqrt( r_old_dot );

    /* p  = z */
    Kokkos::deep_copy( p, z ); Kokkos::fence();

    int k = 0;

    while ( tolerance < norm_res && k < N ) { 
        /* alpha = (r'*z)/(p'*A*p) */
        // Ap = A * p
        KokkosSparse::spmv( "N", one, A, p, zero, Ap); Kokkos::fence();
        // pAp_dot = p'*A*p
		double pAp_dot = KokkosBlas::dot(p, Ap); Kokkos::fence();
        // alpha = r_old_dot/pAp_dot
        double alpha   = r_old_dot / pAp_dot;

        /* x = x + alpha*p */
        /* r = r - alpha*A*p */
        KokkosBlas::axpy(alpha, p, x);   Kokkos::fence();
        KokkosBlas::axpy(-alpha, Ap, r); Kokkos::fence();

        if (prec == 1) {
          // EXERCISE: z = inv(M)*r or solve M*z = r for z or solve (L*U)*z = r for z
          // EXERCISE hint: solve L*Linvr = r     for Linvr, then
          //                solve U*z     = Linvr for z
          //                KokkosSparse::Experimental::sptrsv_solve(...)

        }
        else {
          Kokkos::deep_copy( z, r );
        }

        /* beta = (r'*z)/(r_old'*z_old) */
        /* r_dot = <r,z> */
        /* beta = r_dot/r_old_dot */
        // r_dot = r'*r
        double r_dot = KokkosBlas::dot(r,z); Kokkos::fence();
        double beta  = r_dot / r_old_dot ;

        /* p = z + beta * p */
        KokkosBlas::axpby(one, z, beta, p); Kokkos::fence();

        norm_res = std::sqrt( r_old_dot = r_dot );

        k++;
    }

    Kokkos::fence();

    gettimeofday( &end, NULL );

    // Calculate time
    double time = 1.0 *    ( end.tv_sec  - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );


    // Print results (problem size, time, number of iterations and norm residual)
    printf( "    Results: N (%d), time (%g s), iterations (%d), norm_res(%.20lf)\n", N, time, k, norm_res );
  }

  Kokkos::finalize();

  return 0;
}
