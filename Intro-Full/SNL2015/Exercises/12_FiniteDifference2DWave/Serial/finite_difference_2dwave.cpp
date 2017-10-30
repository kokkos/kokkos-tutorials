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

// -*- C++ -*-
// FiniteDifference2DWave.cc
// An exercise for getting to know Kokkos.
// Here we solve the wave equation in 2d with finite difference

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <chrono>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

int main(int argc, char** argv) {

  // ===============================================================
  // ********************** < inputs> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // change the numberOfCellsPerSide to control the amount
  //  of work done.
  unsigned int numberOfCellsPerSide                = 16 * 100;
  // the courant number determines how much simulation time is
  //  done by each timestep.  don't use higher than 1.
  const double courant                             = 1 / std::sqrt(2);
  // the number of simulation timesteps
  unsigned int numberOfTimesteps                   = 100;
  // the number of output files, which determines how many simulation
  //  timesteps are performed between file writes.
  const unsigned int numberOfOutputFiles           = 100;
  const unsigned int numberOfRenderingCellsPerSide = 50;

  // Read command line arguments
  for(int i=0; i<argc; i++) {
           if( (strcmp(argv[i], "-n") == 0) || (strcmp(argv[i], "-num_cells") == 0)) {
      numberOfCellsPerSide = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-t") == 0) || (strcmp(argv[i], "-time_steps") == 0)) {
      numberOfTimesteps = atof(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("FiniteDifference 2D Wave Options:\n");
      printf("  -num_cells (-n)  <int>: number of cells per side (default: 1600)\n");
      printf("  -time_steps (-t) <int>: number of timesteps (default: 100s)\n");
      printf("  -help (-h):             print this message\n");
    }
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </inputs> ******************************
  // ===============================================================

  const unsigned int fileWriteTimestepInterval =
    std::max((unsigned)1,
             numberOfTimesteps / numberOfOutputFiles);
  const unsigned int heartbeatOutputTimestepInterval =
    std::max((unsigned)1,
             numberOfTimesteps / 10);
  const double courant2 = courant * courant;

  const unsigned int paddedCellsPerSide = numberOfCellsPerSide + 2;

  // initialize the data
  const unsigned int numberOfPaddedCells =
    paddedCellsPerSide * paddedCellsPerSide;
  array<double*, 2> u;
  u[0] = new double[numberOfPaddedCells];
  u[1] = new double[numberOfPaddedCells];
  const std::array<double, 2> gaussianCenter =
    {{paddedCellsPerSide / 2., paddedCellsPerSide / 2.}};
  double maxValue = std::numeric_limits<double>::lowest();
  for (unsigned int i = 0; i < paddedCellsPerSide; ++i) {
    for (unsigned int j = 0; j < paddedCellsPerSide; ++j) {
      const std::array<double, 2> diff =
        {{(i + 0.5) - gaussianCenter[0], (j + 0.5) - gaussianCenter[1]}};
      const double r = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
      const double c = numberOfCellsPerSide / 10;
      const double b = numberOfCellsPerSide / 10;
      const double value = std::exp(-1.*(r-b)*(r-b)/(2*c*c));
      u[0][i * paddedCellsPerSide + j] = value;
      u[1][i * paddedCellsPerSide + j] = value;
      maxValue = std::max(maxValue, value);
    }
  }
  printf("memory size is %8.2e bytes, max initial value is %lf\n",
         float(sizeof(double) * paddedCellsPerSide * paddedCellsPerSide * 2),
         maxValue);

  double totalCalculationTime = 0;
  // for each time step
  unsigned int fileIndex = 0;
  for (unsigned int timestepIndex = 0;
       timestepIndex < numberOfTimesteps; ++timestepIndex) {

    if (timestepIndex % heartbeatOutputTimestepInterval == 0) {
      printf("simulation on timestep %4u/%4u (%%%5.1f)\n",
             timestepIndex, numberOfTimesteps,
             100. * timestepIndex / float(numberOfTimesteps));
    }

    const high_resolution_clock::time_point thisTimestepsTic =
      high_resolution_clock::now();

    const unsigned int t   = timestepIndex % 2;
    const unsigned int tp1 = (timestepIndex + 1) % 2;
    for (unsigned int ii = 0; ii < numberOfCellsPerSide; ++ii) {
      const unsigned int i = ii + 1;
      for (unsigned int jj = 0; jj < numberOfCellsPerSide; ++jj) {
        const unsigned int j = jj + 1;

        const double utij = u[t][i * paddedCellsPerSide + j];
        u[tp1][i * paddedCellsPerSide + j] =
          (2 - 4 * courant2) * utij
          - u[tp1][i * paddedCellsPerSide + j]
          + courant2 * (1*u[t][(i+1) * paddedCellsPerSide +  j   ]
                        + u[t][(i-1) * paddedCellsPerSide +  j   ]
                        + u[t][ i    * paddedCellsPerSide + (j+1)]
                        + u[t][ i    * paddedCellsPerSide + (j-1)]);
      }
    }

    const high_resolution_clock::time_point thisTimestepsToc =
      high_resolution_clock::now();
    const double thisTimestepsElapsedTime =
      duration_cast<duration<double> >(thisTimestepsToc - thisTimestepsTic).count();
    totalCalculationTime += thisTimestepsElapsedTime;

    if (timestepIndex % fileWriteTimestepInterval == 0) {
      // write a file
      char sprintfBuffer[500];
      sprintf(sprintfBuffer, "Serial_%03u.csv", fileIndex);
      const unsigned int t   = timestepIndex % 2;
      FILE* file = fopen(sprintfBuffer, "w");
      const unsigned int interval =
        std::max((unsigned)1,
                 paddedCellsPerSide / numberOfRenderingCellsPerSide);
      double maxValue = std::numeric_limits<double>::lowest();
      for (unsigned int i = 0; i < paddedCellsPerSide; i += interval) {
        fprintf(file, "%4.2f", u[t][i * paddedCellsPerSide + 0]);
        for (unsigned int j = interval; j < paddedCellsPerSide; j += interval) {
          const double value = u[t][i * paddedCellsPerSide + j];
          fprintf(file, ", %4.2f", value);
          maxValue = std::max(maxValue, value);
        }
        fprintf(file, "\n");
      }
      fclose(file);
      printf("max value for file %3u is %4.2lf\n", fileIndex, maxValue);
      ++fileIndex;
    }

  }

  delete[] u[0];
  delete[] u[1];

  printf("total calculation time was %.2lf\n", 1000 * totalCalculationTime);
  return 0;
}
