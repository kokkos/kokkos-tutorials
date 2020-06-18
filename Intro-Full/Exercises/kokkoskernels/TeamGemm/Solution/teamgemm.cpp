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

int main(int argc, char* argv[])
{
  dims_t A_dims, B_dims, C_dims;
  float user_alpha = 1.0;
  float user_beta = 2.0;

  A_dims.n = A_dims.m = 1 << 10;
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
      alpha = atof( argv[i]);
      printf("  User alpha %f\n", alpha);
    }
    if (strcmp(argv[i], "-beta") == 0) {
      beta = atof( argv[i]);
      printf("  User beta %f\n", beta);
    }

    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("  TeamGemm Options:\n" );
      printf("  -A.m <int>:      number of rows in A (default: 1024)\n");
      printf("  -A.n <int>:      number of columns in A (default: 1024)\n");
      printf("  -B.m <int>:      number of rows in B (default: 1024)\n");
      printf("  -B.n <int>:      number of columns in B (default: 1024)\n");
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

  //using Layout = Kokkos::LayoutLeft;
  using Layout = Kokkos::LayoutRight;

  // using ScalarType = float;
  using ScalarType = double;

  using ViewType = Kokkos::View<ScalarType**, Layout>;

  // Timer products
  struct timeval begin, end;

  double tolerance = 0.0000000001;
  ScalarType alpha = (ScalarType) user_alpha;
  ScalarType beta = (ScalarType) user_beta;

  // TODO: Allocate random matrices on the device

  gettimeofday(&begin, NULL);

  // TODO: Start TeamGemm
  Kokkos::fence();

  gettimeofday( &end, NULL );

  // Calculate time
  double time = 1.0 *    (end.tv_sec  - begin.tv_sec) +
                1.0e-6 * (end.tv_usec - begin.tv_usec);

  // TODO: Check result
  
  // Print results (problem size, time).
  printf( "    Results: ( C:%dx%d, A:%dx%d, B:%dx%d, beta:%lf, alpha:%lf ), time( %g s )\n",
          C_dims.m, C_dims.n, A_dims.m, A_dims.n, B_dims.m, B_dims.n, beta, alpha, time);

  Kokkos::finalize();

  return 0;
}
