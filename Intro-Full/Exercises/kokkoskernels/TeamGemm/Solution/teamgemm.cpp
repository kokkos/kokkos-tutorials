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

// EXERCISE Goal: Solve C = beta*C + alpha*A*B using the Batched Team interface for Gemm
//        - KokkosKernels Batched functions

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
#include "Kokkos_Random.hpp"
// EXERCISE: Include header files for proper KokkosKernels Batched functions
// EXERCISE hint: #include "KokkosBatched_Gemm_Decl.hpp" #include "KokkosBatched_Gemm_Team_Impl.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Team_Impl.hpp"

struct dims_t {
  int m, n;
};

void checkDims(dims_t A, dims_t B)
{
  if (A.n != B.m) {
    printf("  A's cols must match B's rows but A:%dx%d and B:%dx%d\n", A.m, A.n, B.m, B.n);
    exit(1);
  }
}

template <class TeamMemberType,
	  class ScalarType,
	  class ViewType,
	  class ATransType,
	  class BTransType,
	  class ViewTypeFilters>
struct functor_TeamGemm {
  ScalarType alpha;
  ViewType A; 
  ViewType B;
  ScalarType beta;
  ViewType C;
  ViewTypeFilters filters;
  const int team_size;
  const int vector_size;

  functor_TeamGemm(ScalarType alpha_,
		   ViewType A_, ViewType B_,
		   ScalarType beta_, ViewType C_,
		   ViewTypeFilters filters_,
		   int team_size_, int vector_size_) :  alpha(alpha_),
						      A(A_),
						      B(B_),
						      beta(beta_),
						      C(C_),
						      filters(filters_),
						      team_size(team_size_),
						      vector_size(vector_size_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMemberType &member) const {
    const int idx = member.league_rank();
    // Fetch 2D sub-matrices
    auto a = Kokkos::subview(A, idx, Kokkos::ALL(), Kokkos::ALL());
    auto b = Kokkos::subview(B, idx, Kokkos::ALL(), Kokkos::ALL());
    auto c = Kokkos::subview(C, idx, Kokkos::ALL(), Kokkos::ALL());
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, team_size), [&](const int &k0) {
	// Fetch 1D column vectors
	auto b_col_vec = Kokkos::subview(a, Kokkos::ALL(), k0);
	auto c_col_vec = Kokkos::subview(c, Kokkos::ALL(), k0);
	auto filter = Kokkos::subview(filters, Kokkos::ALL(), k0);
	Kokkos::parallel_for(Kokkos::TeamThreadRange(member, vector_size), [&](const int &k1) {
	    // Filter each b column vector
	    c_col_vec(k1) *= filter(k1);
	  });
	// Calculate c_col_vec = beta*c_col_vec + alpha*a*b_col_vec  
	KokkosBatched::TeamGemm<TeamMemberType,
				ATransType,
				BTransType,
				KokkosBatched::Algo::Gemm::Unblocked>
	  ::invoke(member, alpha, a, b_col_vec, beta, c_col_vec);
      });
  }
};


int main(int argc, char* argv[])
{
  dims_t A_dims, B_dims, C_dims;
  float user_alpha = 1.0;
  float user_beta = 2.0;

  A_dims.n = A_dims.m = 1 << 5;
  B_dims = C_dims = A_dims;

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-A.m") == 0) {
      A_dims.m = atoi(argv[i]);
      printf("  User A.m is %d\n", A_dims.m);
    }
    if (strcmp(argv[i], "-A.n") == 0) {
      A_dims.n = atoi( argv[i]);
      printf("  User A.n is %d\n", A_dims.n);
    }
    if (strcmp(argv[i], "-B.n") == 0) {
      B_dims.n = atoi( argv[i]);
      printf("  User B.n is %d\n", B_dims.n);
    }
    if (strcmp(argv[i], "-B.n") == 0) {
      B_dims.n = atoi( argv[i]);
      printf("  User B.n is %d\n", B_dims.n);
    }
    if (strcmp(argv[i], "-alpha") == 0) {
      user_alpha = atof( argv[i]);
      printf("  User alpha %f\n", user_alpha);
    }
    if (strcmp(argv[i], "-beta") == 0) {
      user_beta = atof( argv[i]);
      printf("  User beta %f\n", user_beta);
    }

    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("  TeamGemm Options:\n" );
      printf("  -A.m <int>:      number of rows in A (default: 32)\n");
      printf("  -A.n <int>:      number of columns in A (default: 32)\n");
      printf("  -B.m <int>:      number of rows in B (default: 32)\n");
      printf("  -B.n <int>:      number of columns in B (default: 32)\n");
      printf("  -alpha <float>:      value of alpha (default: 1.0)\n");
      printf("  -beta <float>:       value of beta (default: 2.0)\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  // Check dimensions of A and B.
  checkDims(A_dims, B_dims);

  C_dims.m = A_dims.m;
  C_dims.n = B_dims.n;
  
  Kokkos::initialize(argc, argv);

  {
    // Select View Types
    // using ScalarType = float;
    using ScalarType = double;
    //using Layout = Kokkos::LayoutLeft;
    using LayoutType = Kokkos::LayoutRight;
    using DeviceType = Kokkos::Cuda;
    using ViewType = Kokkos::View<ScalarType***, LayoutType, DeviceType>;
    using FilterType = Kokkos::View<ScalarType**, LayoutType, DeviceType>;

    // Timer products
    struct timeval begin, end;

    ScalarType alpha = (ScalarType) user_alpha;
    ScalarType beta = (ScalarType) user_beta;
    const int N = 1 << 7;

    // Allocate A, B, and C matrices on the device
    ViewType A("A", N, A_dims.m, A_dims.n);
    ViewType B("B", N, B_dims.m, B_dims.n);
    ViewType C("C", N, C_dims.m, C_dims.n);
    FilterType filters("filters", B_dims.m, B_dims.n);

    // Populate A, B, and C matrices with random numbers
    using ExecutionSpaceType = DeviceType::execution_space;
    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<ExecutionSpaceType> rand_pool(seed);
    
    Kokkos::fill_random(A, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());
    Kokkos::fill_random(B, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());
    Kokkos::fill_random(C, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());
    Kokkos::fill_random(filters, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());

    gettimeofday(&begin, NULL);

    // Invoke TeamGemm from Vector Loop
    const int num_leagues = N;         /// N teams are formed
    int team_size = C_dims.n;    /// Each team consists of C_dims.n kokkos threads
    int vector_size = C_dims.m;  /// team_size * vector_size concurrent threads are associated within a team

    using TeamMemberType = Kokkos::TeamPolicy<ExecutionSpaceType>::member_type;
    using ATransType = KokkosBatched::Trans::NoTranspose;
    using BTransType = KokkosBatched::Trans::NoTranspose;
    using FunctorType = functor_TeamGemm<TeamMemberType, ScalarType, ViewType, ATransType, BTransType, FilterType>;

    FunctorType functor(alpha, A, B, beta, C, filters, team_size, vector_size);
    Kokkos::TeamPolicy<ExecutionSpaceType> policy(num_leagues, team_size, vector_size);

    Kokkos::parallel_for(policy, functor);
    
    // Wait for the device to return control
    Kokkos::fence();

    gettimeofday(&end, NULL);

    // Calculate time
    double time = 1.0 *    (end.tv_sec  - begin.tv_sec) +
                  1.0e-6 * (end.tv_usec - begin.tv_usec);
    
    // Print results (problem size, time).
    printf( "    Results: ( C:%dx%dx%d, A:%dx%dx%d, B:%dx%dx%d, beta:%lf, alpha:%lf ), time( %g s )\n",
            N, C_dims.m, C_dims.n, N, A_dims.m, A_dims.n, N, B_dims.m, B_dims.n, beta, alpha, time);
  }

  Kokkos::finalize();

  return 0;
}
