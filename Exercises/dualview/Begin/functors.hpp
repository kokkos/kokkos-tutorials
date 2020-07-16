
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

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

typedef Kokkos::DualView<double*> view_type;
const double density_0 = 1;
const double temperature_0 = 300;


template<class ExecutionSpace>
struct ComputePressure {


   static constexpr double gasConstant = 1;

   // If the functor has a public 'execution_space' typedef, that defines
   // the functor's execution space (where it runs in parallel).  This
   // overrides Kokkos' default execution space.

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type::memory_space, view_type::host_mirror_space>::type memory_space;

   // Get the view types on the particular device for which the functor
   // is instantiated.
   //
   // "const_data_type" is a typedef in View (and DualView) which is
   // the const version of the first template parameter of the View.
   // For example, the const_data_type version of double** is const
   // double**.
   // "scalar_array_type" is a typedef in ViewTraits (and DualView) which is the
   // array version of the value(s) stored in the View.
   
   Kokkos::View<view_type::scalar_array_type, view_type::array_layout,
   memory_space> pressure;
   Kokkos::View<view_type::const_data_type, view_type::array_layout,
   memory_space, Kokkos::MemoryRandomAccess> temperature;
   Kokkos::View<view_type::const_data_type, view_type::array_layout,
   memory_space, Kokkos::MemoryRandomAccess> density;

   // Constructor takes DualViews, synchronizes them to the device,
   // then marks them as modified on the device.
   
   ComputePressure (view_type dv_pressure, view_type dv_temperature, view_type dv_density)
   {
      // Extract the view on the correct Device (i.e., the correct
      // memory space) from the DualView.  DualView has a template
      // method, view(), which is templated on the memory space.  If the
      // DualView has a View from that memory space, view() returns the
      // View in that space.
      
      pressure = dv_pressure.template view<memory_space> ();
      density  = dv_density.template view<memory_space> ();
      temperature  = dv_temperature.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      //
      // DualView's sync() method is templated on a memory space, and
      // synchronizes the DualView in a one-way fashion to that memory
      // space.  "Synchronizing" means copying, from the other memory
      // space to the Device memory space.  sync() does _nothing_ if the
      // Views on the two memory spaces are in sync.  DualView
      // determines this by the user manually marking one side or the
      // other as modified; see the modify() call below.

      dv_pressure.sync<memory_space> ();
      dv_temperature.sync<memory_space> ();
      dv_density.sync<memory_space> ();

      // Mark pressure as modified.
      dv_pressure.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      pressure(i) = density(i)*gasConstant*temperature(i);
   }
};


template<class ExecutionSpace>
struct ComputeInternalEnergy {

   static constexpr double C_v = 1;

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type::memory_space, view_type::host_mirror_space>::type memory_space;

   Kokkos::View<view_type::scalar_array_type, view_type::array_layout, memory_space> energy;

   Kokkos::View<view_type::const_data_type, view_type::array_layout, memory_space, Kokkos::MemoryRandomAccess> temperature;

   ComputeInternalEnergy(view_type dv_energy, view_type dv_temperature)
   {
      energy = dv_energy.template view<memory_space> ();
      temperature  = dv_temperature.template view<memory_space> ();

      dv_energy.sync<memory_space> ();
      dv_temperature.sync<memory_space> ();

      // Mark energy as modified
      dv_energy.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      energy(i) = C_v*temperature(i);
   }
};


