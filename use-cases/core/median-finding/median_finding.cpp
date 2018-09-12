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
// Questions Contact  
//
// ************************************************************************
//@HEADER
*/

// The simple model creates n points in 1d evenly distributed 0..1,
// then coordinates are squared so half the points lie below 0.25.
// A thread reduce loop inside a team reduce loop checks each coordinate
// and determines if they are left or right of the current cut.
// The total weights to the left of the cut are reduced (summed).
// Then the cut is shifted to estimate a new cut position,
// where the goal is to have the left and right have the same weight.
// The main loop runs the full test for a range of team counts
// and logs the time cost for each.

#include <Kokkos_Core.hpp>

// store results from a run
struct Result {
  int time_ms;        // time in ms for actual partition
  double cut_line;    // the final cut
  int n_teams;        // how many teams we ran with
};

// in this simple model N points is restricted to
// be a power of 2 and we test N teams in powers
// of 2 as well so it's always an even factor.
#define KOKKOS_N_POINTS_POW_OF_2 23

// main will call this simple_model test for varies n_teams
Result simple_model(int n_teams) {

    // Fix the total number of points
    const int n_points = pow(2,KOKKOS_N_POINTS_POW_OF_2); // keep this in int bounds or change type
    const int n_points_per_team = n_points / n_teams;

    // currently setup assuming n_teams will be a factor
    if(n_points_per_team * n_teams != n_points) {
      throw std::logic_error("For now this assumes n_points is a multiple of n_teams.");
    }

    // for now simple epsilon assumes no duplicate points (should probably have a tiny epsilon for rounding error)
    double epsilon = 0.5; // the final weight should not deviate from the perfect weight by more than 1/2 a weight unit

    // make the views using the default device
    typedef Kokkos::View<double*> view_t;
    view_t x("x", n_points);

    // will fix team count and use Kokkos::AUTO - this policy is the only one used in the model
    auto policy = Kokkos::TeamPolicy<>(n_teams, Kokkos::AUTO);
    typedef typename Kokkos::TeamPolicy<>::member_type member_type;

    // initialize the coordinates in a parallel_for - we want UVM off to work
    Kokkos::parallel_for("initialize points", n_points, KOKKOS_LAMBDA (int i) {
      // purpose is to make a simple shift so coords are not ordered
      // after scaling 0-1 do x^2 to make an uneven distribution
      // that results in the proper cut shift being 0.25 since
      // half the points have i > 0.5 which is i*i = 0.25
      int shift_index = i + n_points/2;
      if(shift_index >= n_points) {
        shift_index -= n_points;
      }                                                                                                                                                       
      double val = (double) (shift_index) / (double) (x.size() - 1);
      x(i) = val * val; // range 0 < 1.0   square it to make non uniform with proper cut at 0.25
    });

    // the target weight is what we expect on the left side (or same for right)
    // this simple model is currently only calculating a single cut point
    // note currently assumes uniform weights so 1.0 for each coordinate
    auto targetWeight = (double) x.size() / 2.0;

    // now we can guess the initial cut_line as a starting point to be 0.5
    // in the current form the true answer is 0.25 because the points were
    // evenly spaced, then squared, so half end up being below 0.25
    double cut_line = 0.5;

    // just for logging, get the time on the main loop
    // currently I'm interested to see how n_teams impacts the total time
    typedef std::chrono::high_resolution_clock Clock;
    auto clock_start = Clock::now();

    // main loop will reduce over threads (inner) and then teams (outer)
    bool bDone = false; // waits until cut is satisfied
    while(!bDone) {
      // we will reduce the total weight to the left of the cut over the teams
      double storeWeightLeft = 0;
      Kokkos::parallel_reduce("main loop", policy, KOKKOS_LAMBDA(member_type teamMember, double & weightLeft) {
        // inner loop reduces the total weight over the threads
        double storeTeamWeightLeft = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, n_points_per_team),
          [=] (int ii, double & teamWeightLeft) {
          int i = ii + teamMember.league_rank() * n_points_per_team;
          if(x(i) < cut_line) {
            teamWeightLeft += 1.0; // for now weights are uniform: 1.0 per coordinates
          }
        }, storeTeamWeightLeft);
        if(teamMember.team_rank() == 0) {
          weightLeft += storeTeamWeightLeft;
        }
      }, storeWeightLeft);

      // now update the cut - estimate assuming uniform weights and uniform distribution of weights
      double storeWeightRight = static_cast<double>(n_points) - storeWeightLeft; // left + right = total
      if(storeWeightLeft < targetWeight - epsilon) { // cut moves right
        // the amount if weight to the right of the cut is storeWeightRight
        // the amount of weight we want to cross shifting right is: targetWeight - storeWeightLeft
        // the total distance on the right side is: 1.0 - cut_line    (all points are 0..1)
        // so we assume the region from cut_line to 1.0 is uniformly weighted
        // the fraction we want to cover is (targetWeight - storeWeightLeft) / storeWeightRight
        // so we take that fraction of the right side distance as our estimated shift
        cut_line += (1.0 - cut_line) * (targetWeight - storeWeightLeft) / storeWeightRight;
      }
      else if(storeWeightLeft > targetWeight + epsilon) { // cut moves left
        // same as above except now we are absorbing a section on the left with total size cut_line
        cut_line -= cut_line * (storeWeightLeft - targetWeight) / storeWeightLeft;
      }
      else {
        bDone = true; // when the cut is close enough we end the main loop
      }
    }

    // set the output values
    Result result;
    result.time_ms = static_cast<int>(std::chrono::duration_cast<
      std::chrono::milliseconds>(Clock::now() - clock_start).count());
    result.cut_line = cut_line;
    result.n_teams = n_teams;
    return result;
}

int main( int argc, char* argv[] )
{
  Kokkos::ScopeGuard kokkosScope(argc, argv); 

  std::vector<Result> results; // store the results for each run
  for(int n_teams = 1; n_teams <= pow(2,KOKKOS_N_POINTS_POW_OF_2); n_teams *=2) {
    Result result = simple_model(n_teams);
    results.push_back(result); // add to vector for logging at end
  }

  // now loop and log each result - shows how n_teams impacts total time
  for(auto&& result : results) {
    printf("teams: %8d   cut: %.2lf    time: %d ms\n",
      result.n_teams, result.cut_line, result.time_ms);
  }
}
