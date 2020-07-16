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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
// 
// ************************************************************************
//@HEADER
 */

#include <ostream>
#include "functors.hpp"

/*
 * Goal:  Understand how to move data back and forth between host and device using DualView
 *
 *  The idea is that this example simulates a more complex code that you might be converting to use
 *  kokkos, and it is not clear which parts of the code are running on the host or the device.
 *
 */
// DualView helps you manage data and computations that take place on
// two different memory spaces.  Examples include CUDA device memory
// and (CPU) host memory (currently implemented), or Intel Knights
// Landing MCDRAM and DRAM (not yet implemented).  For example, if you
// have ported only some parts of you application to run in CUDA,
// DualView can help manage moving data between the parts of your
// application that work best with CUDA, and the parts that work
// better on the CPU.
//
// A DualView takes the same template parameters as a View, but
// contains two Views: One that lives in the DualView's memory space,
// and one that lives in that memory space's host mirror space.  If
// both memory spaces are the same, then the two Views just alias one
// another.  This means that you can use DualView all the time, even
// when not running in a memory space like CUDA.  DualView's
// operations to help you manage memory take almost no time in that
// case.  This makes your code even more performance portable.


void load_state(view_type density, view_type temperature);
void compute_pressure(view_type pressure, view_type density, view_type temperature);
void compute_internal_energy(view_type energy, view_type temperature);
void compute_enthalpy(view_type enthalpy, view_type energy, view_type pressure, view_type density);
void check_results(view_type pressure, view_type energy, view_type enthalpy);

int main (int narg, char* arg[]) {
  
  std::cout << "initializing kokkos....." <<std::endl;

   Kokkos::initialize (narg, arg);
   
   std::cout << "......done." << std::endl;
   {
      // Create DualViews. This will allocate on both the device and its
      // host_mirror_device.

      const int size = 1000000;

      view_type pressure ("pressure",size);
      view_type density ("density",size);
      view_type temperature ("temperature",size);
      view_type energy ("energy",size);
      view_type enthalpy ("enthalpy",size);
      
      load_state(density, temperature);

      // this section of code is supposed to mimic the structure of a time loop in a
      // more complex physics app
      const size_t maxSteps = 1;
      for (size_t step = 0; step < maxSteps; ++step) {
	compute_pressure(pressure, density, temperature);
	compute_internal_energy(energy, temperature);
	compute_enthalpy(enthalpy, energy, pressure, density);
      }

      check_results(pressure, energy, enthalpy);

   }

   Kokkos::finalize();
}
void load_state(view_type density, view_type temperature) {

   // Get a reference to the host view directly (equivalent to
   // density.view<view_type::host_mirror_space>() )

   view_type::t_host h_density = density.h_view;
   view_type::t_host h_temperature = temperature.h_view;

   for (view_type::size_type j = 0; j < h_density.extent(0); ++j) {
      h_density(j) = density_0;
      h_temperature(j) = temperature_0;
   }
   // Mark as modified on the host_mirror_space so that a
   // sync to the device will actually move data.

   density.modify<view_type::host_mirror_space> ();
   temperature.modify<view_type::host_mirror_space> ();
}

void compute_pressure(view_type pressure, view_type density, view_type temperature) {

   // Run on the device.  This will cause data movement to the device,
   // since it was marked as modified on the host.

   const int size = pressure.extent(0);

   Kokkos::parallel_for(size, ComputePressure<view_type::execution_space>(pressure, temperature, density));
   Kokkos::fence();
}

void compute_internal_energy(view_type energy, view_type temperature) {

   const int size = energy.extent(0);
   Kokkos::parallel_for(size, ComputeInternalEnergy<view_type::execution_space>(energy, temperature));
   Kokkos::fence();
}

void compute_enthalpy(view_type enthalpy, view_type energy, view_type pressure, view_type density) {

   const int size = enthalpy.extent(0);

   Kokkos::parallel_for(size, ComputeEnthalpy<view_type::execution_space>(enthalpy, energy, pressure, density));
   Kokkos::fence();

}
void check_results(view_type dv_pressure, view_type dv_energy, view_type dv_enthalpy) {

   const double R = ComputePressure<view_type::host_mirror_space>::gasConstant;
   const double thePressure =  R*density_0*temperature_0;

   const double cv = ComputeInternalEnergy<view_type::host_mirror_space>::C_v;
   const double theEnergy = cv*temperature_0;

   const double theEnthalpy = theEnergy + thePressure/density_0;

   auto pressure  = dv_pressure.h_view;
   auto energy = dv_energy.h_view;
   auto enthalpy  = dv_enthalpy.h_view;

   dv_pressure.sync<view_type::host_mirror_space> ();
   dv_energy.sync<view_type::host_mirror_space> ();
   dv_enthalpy.sync<view_type::host_mirror_space> ();

   double pressureError = 0;
   double energyError = 0;
   double enthalpyError = 0;
   const int size = energy.extent(0);
   for(int i = 0; i < size; ++i) {
      pressureError += (pressure(i) - thePressure)*(pressure(i) - thePressure);
      energyError += (energy(i) - theEnergy)*(energy(i) - theEnergy);
      enthalpyError += (enthalpy(i) - theEnthalpy)*(enthalpy(i) - theEnthalpy);
   }

   std::cout << "pressure error = " << pressureError << std::endl;
   std::cout << "energy error = " << energyError << std::endl;
   std::cout << "enthalpy error = " << enthalpyError << std::endl;

}


