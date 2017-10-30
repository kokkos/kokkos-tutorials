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


int main(int argc, char* argv[]) {

  // Setting default parameters
  int num_points = 100000;  // number of vectors
  int num_centroids  = 100; // length of vectors
  int max_iter = 500;       // maximum number of iterations of the test
  int stride_c_sum = 3;
  int stride_c_count = 1;

  // Read command line arguments
  for(int i=0; i<argc; i++) {
           if( (strcmp(argv[i], "-p") == 0) || (strcmp(argv[i], "-num_points") == 0)) {
      num_points = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-c") == 0) || (strcmp(argv[i], "-num_centroids") == 0)) {
      num_centroids = atof(argv[++i]);
    } else if( (strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "-max_iter") == 0)) {
      max_iter = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-pad") == 0)) {
      int stride_c_sum = 8;
      int stride_c_count = 16;
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("Scalar Integration Otpions:\n");
      printf("  -num_vectors (-v)  <int>: number of vectors (default: 1000)\n");
      printf("  -length (-l) <int>:       vector length (default: 10000)\n");
      printf("  -max_iter (-i) <int>:     maximum number of iterations (default: 500)\n");
      printf("  -pad (-p):                pad arrays for performant atomic access\n");
      printf("  -help (-h):               print this message\n");
    }
  }

  // Allocate data arrays
  double* points = new double[num_points*3]; // the point coordinates
  int* points_closest_centroid = new int[num_points]; // index of closest centroid for a point
  double* centroids = new double[num_centroids*3]; // the centroid coordinates
  double* centroids_sum = new double[num_centroids * stride_c_sum]; // sum of point contributions to a centroid
  int* centroids_count = new int[num_centroids * stride_c_count]; // count of points contributing to a centroid

  srand(90391);  

  // Initialize points with random coordinates in a 1Mx1Mx1M grid (do not parallelize to get same answer)
  for(int i = 0; i < num_points; i++) {
    points[i*3 + 0] = rand()%(1024*1024);
    points[i*3 + 1] = rand()%(1024*1024);
    points[i*3 + 2] = rand()%(1024*1024);
    points_closest_centroid[i] = -1;
  }

  // Initialize centroids
  #pragma omp parallel for
  for(int i = 0; i < num_centroids; i++) {
    centroids_sum[i*stride_c_sum + 0] = points[i*3 + 0];
    centroids_sum[i*stride_c_sum + 1] = points[i*3 + 1];
    centroids_sum[i*stride_c_sum + 2] = points[i*3 + 2];
    centroids_count[i*stride_c_count] = 1;
  }

  // Time finding of centroids
  struct timeval begin,end;
  int changed = 1;

  gettimeofday(&begin,NULL);

  // iterate
  int iter = 0;
  while( (iter < max_iter) && (changed > 0)) {
    changed = 0;
    #pragma omp parallel for
    for(int i = 0; i < num_centroids; i++) {
      const double inv_count = 1.0/centroids_count[i];
      centroids[i*3 + 0] = centroids_sum[i*stride_c_sum + 0] * inv_count;
      centroids[i*3 + 1] = centroids_sum[i*stride_c_sum + 1] * inv_count;
      centroids[i*3 + 2] = centroids_sum[i*stride_c_sum + 2] * inv_count;
      centroids_sum[i*stride_c_sum + 0] = 0.0;
      centroids_sum[i*stride_c_sum + 1] = 0.0;
      centroids_sum[i*stride_c_sum + 2] = 0.0;
      centroids_count[i*stride_c_count] = 0;
    }

    #pragma omp parallel for reduction(+: changed)
    for(int i = 0; i < num_points; i++) {
      const double x = points[i*3 + 0];
      const double y = points[i*3 + 1];
      const double z = points[i*3 + 2];
      double min_dist2 = 1e20;
      int min_indx = -1;
      for(int j = 0; j < num_centroids; j++) {
        const double dx = centroids[j*3 + 0] - x;
        const double dy = centroids[j*3 + 1] - y;
        const double dz = centroids[j*3 + 2] - z;
        const double dist2 = dx*dx + dy*dy + dz*dz;
        if(dist2 < min_dist2) {
          min_dist2 = dist2;
          min_indx = j;
        }
      }
      #pragma omp atomic update
      centroids_sum[min_indx*stride_c_sum + 0] += x;
      #pragma omp atomic update
      centroids_sum[min_indx*stride_c_sum + 1] += y;
      #pragma omp atomic update
      centroids_sum[min_indx*stride_c_sum + 2] += z;
      #pragma omp atomic update
      centroids_count[min_indx*stride_c_count]++;
      
      if(points_closest_centroid[i] != min_indx) {
        changed++;
        points_closest_centroid[i] = min_indx;
      }
    }
    printf("%i Changed: %i of %i\n",iter,changed,num_points);
    iter++;
  }


  gettimeofday(&end,NULL);

  // Calculate time
  double time = 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);

  // Error check
  int error = 0;

  // Print time

  if(error==0) { 
    printf("%i %i %lf %e\n",num_points,num_centroids,time,time/iter);
  }
  else printf("Error\n");

}
