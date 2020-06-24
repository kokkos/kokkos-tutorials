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

// EXERCISE Goal: Get familiar with KokkosKernels interfaces of incomplete LU factorization ILU(k) for CRS matrices: 
//        - Create handle
//        - Perform symbolic phase: determines the nonzero patterns of L and U, and does level scheduling
//        - Perform numeric phase: uses the nonzero patterns in the symbolic phase to perform an ILU factorization on the original matrix

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
// EXERCISE: Include header files for proper KokkosKernels BLAS and Sparse Matrix functions
// EXERCISE hint: KokkosBlas1_nrm2.hpp, KokkosSparse_CrsMatrix.hpp, KokkosSparse_spmv.hpp, KokkosSparse_spiluk.hpp

#include <KokkosKernels_IOUtils.hpp>
#include "KokkosKernels_Test_Structured_Matrix.hpp"

#define EXPAND_FACT 6 // a factor used in expected sizes of L and U

enum {DEFAULT, LVLSCHED_RP, LVLSCHED_TP1};

int main( int argc, char* argv[] )
{
  using scalar_t  = double;
  using lno_t     = int;
  using size_type = int;

  int nx = 10;                // grid points in 'x' direction
  int ny = 10;                // grid points in 'y' direction
  int test_algo = LVLSCHED_RP;// kernel implementation
  int k = 0;                  // fill level
  int team_size = -1;         // team size
  
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
    else if ( strcmp( argv[ i ], "-algo" ) == 0 ) {
      i++;
      if ( strcmp( argv[ i ], "lvlrp" ) == 0 ) {
        test_algo = LVLSCHED_RP;
      }
      if ( strcmp( argv[ i ], "lvltp1" ) == 0 ) {
        test_algo = LVLSCHED_TP1;
      }
    }
    else if ( strcmp( argv[ i ], "-ts" ) == 0 ) {
      team_size = atoi( argv[ ++i ] );
      printf( "  User's team_size is %d\n", team_size );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  ILU(k) Options: (simple Laplacian matrix on a cartesian grid where nrows = nx * ny) \n" );
      printf( "  -nx <int>     : grid points in x direction (default: 10)\n" );
      printf( "  -ny <int>     : grid points in y direction (default: 10)\n" );
      printf( "  -k <int>      : Fill level (default: 0)\n" );
      printf( "  -algo [OPTION]: Kernel implementation (default: lvlrp)\n" );
      printf( "                  [OPTIONS]: lvlrp, lvltp1\n");
      printf( "  -ts <int>     : team size only when lvltp1 is used (default: -1)'.\n");
      printf( "  -help (-h)    : Print this message\n\n" );
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
    using lno_view_t      = typename graph_t::row_map_type::non_const_type;//row map view type
    using lno_nnz_view_t  = typename graph_t::entries_type::non_const_type;//entries view type
    using scalar_view_t   = typename crsmat_t::values_type::non_const_type;//values view type
    using KernelHandle    = KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
                                                                              execution_space, memory_space, memory_space>;

    std::cout << "ILU(k) execution_space: " << typeid(execution_space).name() << ", memory_space: " << typeid(memory_space).name() << std::endl;

    // Timer products
    struct timeval begin, end;

    //
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

    // Create SPILUK handle
    KernelHandle kh;

    // EXERCISE: Create a SPILUK handle
    // EXERCISE hint: input arguments include implementation type (i.e. KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP or KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1), number of matrix rows, expected nnz of L and expected nnz of U
    std::cout << "Create handle ..." << std::endl;
    switch(test_algo) {
      case LVLSCHED_RP: //Using range policy (KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP)
        // EXERCISE hint: kh.create_spiluk_handle(implementation_type, number_of_matrix_rows, expected_nnz_of_L, expected_nnz_of_U)
        
        std::cout << "Kernel implementation type: "; kh.get_spiluk_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1: //Using team policy (KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1)
        // EXERCISE hint: kh.create_spiluk_handle(implementation_type, number_of_matrix_rows, expected_nnz_of_L, expected_nnz_of_U)

        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh.get_spiluk_handle()->set_team_size(team_size);
        std::cout << "Kernel implementation type: "; kh.get_spiluk_handle()->print_algorithm();
        break;
      default: //Using range policy
        // EXERCISE hint: kh.create_spiluk_handle(implementation_type, number_of_matrix_rows, expected_nnz_of_L, expected_nnz_of_U)

        std::cout << "Kernel implementation type: "; kh.get_spiluk_handle()->print_algorithm();
    }

    auto spiluk_handle = kh.get_spiluk_handle();

    // EXERCISE: Allocate row map, entries, values views (1-D) for L and U
    // EXERCISE hint: get the number of nonzero entries of L and U by spiluk_handle->get_nnzL() and spiluk_handle->get_nnzU()
    //                row map view type: lno_view_t
    //                entries view type: lno_nnz_view_t
    //                values view type:  scalar_view_t


    std::cout << "Expected L_row_map size = " << L_row_map.extent(0) << std::endl;
    std::cout << "Expected L_entries size = " << L_entries.extent(0) << std::endl;	
    std::cout << "Expected L_values size  = " << L_values.extent(0)  << std::endl;	
    std::cout << "Expected U_row_map size = " << U_row_map.extent(0) << std::endl;	
    std::cout << "Expected U_entries size = " << U_entries.extent(0) << std::endl;	
    std::cout << "Expected U_values size  = " << U_values.extent(0)  << std::endl << std::endl;	

    // Symbolic phase
    std::cout << "Run symbolic phase ..." << std::endl;
    gettimeofday( &begin, NULL );
    // EXERCISE: Run symbolic phase
    // EXERCISE hint: KokkosSparse::Experimental::spiluk_symbolic(...)
    //                See https://github.com/kokkos/kokkos-kernels/wiki/SPARSE%3A%3Aspiluk for details

    Kokkos::fence();
    gettimeofday( &end, NULL );
    std::cout << "ILU(" << fill_lev << ") Symbolic Time: " << 1.0 * ( end.tv_sec  - begin.tv_sec  ) +
                                                           1.0e-6 * ( end.tv_usec - begin.tv_usec ) << " seconds" << std::endl;

    // EXERCISE: Resize L and U to their actual sizes
    // EXERCISE hint: use Kokkos::resize(...) and spiluk_handle->get_nnzL(), spiluk_handle->get_nnzU() for their actual sizes	


    // Check
    std::cout << "Actual L_row_map size = " << L_row_map.extent(0) << std::endl;
    std::cout << "Actual L_entries size = " << L_entries.extent(0) << std::endl;
    std::cout << "Actual L_values size  = " << L_values.extent(0)  << std::endl;
    std::cout << "Actual U_row_map size = " << U_row_map.extent(0) << std::endl;
    std::cout << "Actual U_entries size = " << U_entries.extent(0) << std::endl;
    std::cout << "Actual U_values size  = " << U_values.extent(0)  << std::endl;
    std::cout << "ILU(k) fill_level: "   << fill_lev << std::endl;
    std::cout << "ILU(k) fill-factor: "  << (spiluk_handle->get_nnzL() + spiluk_handle->get_nnzU() - N)/(double)nnzA << std::endl;
    std::cout << "num levels: "          << spiluk_handle->get_num_levels() << std::endl;
    std::cout << "max num rows levels: " << spiluk_handle->get_level_maxrows() << std::endl << std::endl;

    // Numeric phase
    std::cout << "Run numeric phase ..." << std::endl;
    gettimeofday( &begin, NULL );
    // EXERCISE: Run numeric phase
    // EXERCISE hint: KokkosSparse::Experimental::spiluk_numeric(...)
    //                See https://github.com/kokkos/kokkos-kernels/wiki/SPARSE%3A%3Aspiluk for details

    Kokkos::fence();
    gettimeofday( &end, NULL );
    std::cout << "ILU(" << fill_lev << ") Numeric Time: " << 1.0 * ( end.tv_sec  - begin.tv_sec  ) +
                                                          1.0e-6 * ( end.tv_usec - begin.tv_usec ) << " seconds" << std::endl;

    // Compute row-sum difference: A*e-L*U*e
    crsmat_t L("L", N, N, spiluk_handle->get_nnzL(), L_values, L_row_map, L_entries);
    crsmat_t U("U", N, N, spiluk_handle->get_nnzU(), U_values, U_row_map, U_entries);
    ViewVectorType e_one  ( "e_one",  N );
    ViewVectorType bb     ( "bb",     N );
    ViewVectorType bb_tmp ( "bb_tmp", N );
    
    Kokkos::deep_copy( e_one, scalar_t(1) );
    
    KokkosSparse::spmv( "N", one, A, e_one,  zero, bb);
    KokkosSparse::spmv( "N", one, U, e_one,  zero, bb_tmp);
    KokkosSparse::spmv( "N", one, L, bb_tmp, mone, bb);

    double bb_nrm = KokkosBlas::nrm2(bb);

    std::cout << "Row-sum difference: nrm2(A*e-L*U*e) = " << std::setprecision(15) << bb_nrm << std::endl;

    kh.destroy_spiluk_handle();
  }

  Kokkos::finalize();

  return 0;
}
