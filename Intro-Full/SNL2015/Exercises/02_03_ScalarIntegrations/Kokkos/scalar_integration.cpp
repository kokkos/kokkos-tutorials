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

#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {

  // Parameters
  double x_start = 0.0; // integration start
  double x_end = 2.0;   // integration end
  int num_intervals = 10000000; // integration intervals
  int nrepeat = 10;     // number of integration invocations

  // Read command line arguments
  for(int i=0; i<argc; i++) {
           if( strcmp(argv[i], "-n") == 0) {
      num_intervals = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-x_start") == 0) {
      x_start = atof(argv[++i]);
    } else if( strcmp(argv[i], "-x_end") == 0) {
      x_end = atof(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("Scalar Integration Otpions:\n");
      printf("  -n <int>:         number of integration intervals (default: 100000000)\n");
      printf("  -x_start <float>: begin of integration range (default: 1.0)\n");
      printf("  -x_end <float>:   end of integration range (default: 2.0)\n");
      printf("  -nrepeat <int>:   number of integration invocations (default: 10)\n");
      printf("  -help (-h):       print this message\n");
    }
  }

  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  // Time force computation
  struct timeval begin,end;

  gettimeofday(&begin,NULL);

  for(int k = 0; k < nrepeat; k++) {

    // Integrate sin(x) from x=x_start to x=x_end;
    double integral = 0.0;
    Kokkos::parallel_reduce(num_intervals, KOKKOS_LAMBDA (const int& i, double& lsum) {
      const double x = x_start + 1.0 * i / num_intervals * (x_end-x_start);
      lsum += sin(x);
    },integral);
    integral *= (x_end-x_start)/num_intervals;

    // Check correct result (this will fail for less than 10M integration intervalls
    double diff = integral - (-cos(x_end)+cos(x_start));
    if(diff*diff > 1e-14) printf("Error %e %lf %lf\n",diff,integral,(cos(x_start)-cos(x_end)));
  }
  gettimeofday(&end,NULL);

  // Print results
  double time = 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);

  printf("NumIntervals time FOM\n");
  printf("%i %lf %e\n",num_intervals,time,1.0e-6*num_intervals*nrepeat/time);

  //Finalize Kokkos
  Kokkos::finalize();

}
