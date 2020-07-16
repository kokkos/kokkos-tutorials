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

// EXERCISE Goal: Demonstrate usage of TeamGemm interface
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

// Create a functor for running TeamGemm on the device
template <class TeamMemberType,
	        class ScalarType,
	        class ViewType,
	        class ATransType,
	        class BTransType>
struct TeamGemmFunctor {
  ScalarType alpha;
  ViewType A; 
  ViewType B;
  ScalarType beta;
  ViewType C;
  const int team_size;

  TeamGemmFunctor(ScalarType alpha_,
		              ViewType A_, ViewType B_,
		              ScalarType beta_, ViewType C_, 
                  int team_size_) : alpha(alpha_),
						      A(A_),
						      B(B_),
						      beta(beta_),
						      C(C_),
						      team_size(team_size_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMemberType &member) const {
    // Fetch the index of the calling team within the league
    const int team_idx = member.league_rank();

    // Fetch 2D sub-matrices for this league
    auto a = Kokkos::subview(A, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto b = Kokkos::subview(B, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto c = Kokkos::subview(C, team_idx, Kokkos::ALL(), Kokkos::ALL());

    // Calculate c = beta*c + alpha*a*b, using all threads in this league
    KokkosBatched::TeamGemm<TeamMemberType,
                            ATransType,
                            BTransType,
                            KokkosBatched::Algo::Gemm::Unblocked>
                  ::invoke(member, alpha, a, b, beta, c);
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
      A_dims.m = atoi(argv[++i]);
      printf("  User A.m is %d\n", A_dims.m);
    }
    if (strcmp(argv[i], "-A.n") == 0) {
      A_dims.n = atoi(argv[++i]);
      printf("  User A.n is %d\n", A_dims.n);
    }
    if (strcmp(argv[i], "-B.m") == 0) {
      B_dims.m = atoi(argv[++i]);
      printf("  User B.m is %d\n", B_dims.m);
    }
    if (strcmp(argv[i], "-B.n") == 0) {
      B_dims.n = atoi(argv[++i]);
      printf("  User B.n is %d\n", B_dims.n);
    }
    if (strcmp(argv[i], "-alpha") == 0) {
      user_alpha = atof(argv[++i]);
      printf("  User alpha %f\n", user_alpha);
    }
    if (strcmp(argv[i], "-beta") == 0) {
      user_beta = atof(argv[++i]);
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

    // Timer products
    struct timeval begin, end;

    ScalarType alpha = (ScalarType) user_alpha;
    ScalarType beta = (ScalarType) user_beta;
    const int N = 1 << 7;

    // Allocate A, B, and C matrices on the device
    ViewType A("A", N, A_dims.m, A_dims.n);
    ViewType B("B", N, B_dims.m, B_dims.n);
    ViewType C("C", N, C_dims.m, C_dims.n);

    // Populate A, B, and C matrices with random numbers
    using ExecutionSpaceType = DeviceType::execution_space;
    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<ExecutionSpaceType> rand_pool(seed);
    
    Kokkos::fill_random(A, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());
    Kokkos::fill_random(B, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());
    Kokkos::fill_random(C, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<ExecutionSpaceType>, ScalarType>::max());

    gettimeofday(&begin, NULL);

    const int num_leagues = N;         /// N teams are formed
    int team_size = C_dims.m;          /// team_size concurrent threads are associated within a team

    using TeamMemberType = Kokkos::TeamPolicy<ExecutionSpaceType>::member_type;
    using ATransType = KokkosBatched::Trans::NoTranspose;
    using BTransType = KokkosBatched::Trans::NoTranspose;
    using FunctorType = TeamGemmFunctor<TeamMemberType, ScalarType, ViewType, ATransType, BTransType>;

    FunctorType functor(alpha, A, B, beta, C, team_size);
    Kokkos::TeamPolicy<ExecutionSpaceType> policy(num_leagues, team_size);

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
