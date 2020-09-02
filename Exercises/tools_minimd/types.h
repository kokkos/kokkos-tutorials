/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#ifndef TYPES_H
#define TYPES_H

#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
enum ForceStyle {FORCELJ, FORCEEAM};

#ifndef MAX_STACK_TYPES
#define MAX_STACK_TYPES 3
#endif

#ifndef PRECISION
#define PRECISION 2
#endif
#if PRECISION==1
typedef float MMD_float;
#else
typedef double MMD_float;
#endif
typedef int MMD_int;
typedef int MMD_bigint;


#ifndef PAD4
#define PAD 3
#else
#define PAD 4
#endif

#ifdef __INTEL_COMPILER
#ifndef ALIGNMALLOC
#define ALIGNMALLOC 64
#endif
#define RESTRICT __restrict
#endif


#ifndef RESTRICT
#define RESTRICT
#endif

typedef Kokkos::DefaultExecutionSpace DeviceType;
typedef Kokkos::HostSpace::execution_space HostType;

typedef Kokkos::DualView<MMD_float*[PAD],Kokkos::LayoutRight> x_dual_view_type;
typedef Kokkos::DualView<MMD_float*> float_1d_dual_view_type;
typedef Kokkos::DualView<MMD_float**> float_2d_dual_view_type;
typedef Kokkos::DualView<MMD_int*> int_1d_dual_view_type;
typedef Kokkos::DualView<MMD_int**> int_2d_dual_view_type;
typedef Kokkos::DualView<MMD_int> int_dual_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight> x_view_type;
typedef Kokkos::View<MMD_float**> float_2d_view_type;
typedef Kokkos::View<MMD_float*> float_1d_view_type;
typedef Kokkos::View<MMD_int*> int_1d_view_type;
typedef Kokkos::View<MMD_int**> int_2d_view_type;
typedef Kokkos::View<MMD_int**,Kokkos::LayoutRight> int_2d_lr_view_type;
typedef Kokkos::View<MMD_int> int_view_type;

typedef Kokkos::View<const MMD_float*[PAD],Kokkos::LayoutRight> x_const_view_type;
typedef Kokkos::View<const MMD_int*> int_1d_const_view_type;
typedef Kokkos::View<const MMD_int**> int_2d_const_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Atomic> > x_atomic_view_type;
typedef Kokkos::View<MMD_float**,Kokkos::MemoryTraits<Kokkos::Atomic> > float_2d_atomic_view_type;
typedef Kokkos::View<MMD_float*,Kokkos::MemoryTraits<Kokkos::Atomic> > float_1d_atomic_view_type;
typedef Kokkos::View<MMD_float,Kokkos::MemoryTraits<Kokkos::Atomic> > float_atomic_view_type;
typedef Kokkos::View<MMD_int*,Kokkos::MemoryTraits<Kokkos::Atomic> > int_1d_atomic_view_type;
typedef Kokkos::View<MMD_int**,Kokkos::MemoryTraits<Kokkos::Atomic> > int_2d_atomic_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > x_atomic_um_view_type;
typedef Kokkos::View<MMD_float*,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > float_1d_atomic_um_view_type;

typedef Kokkos::View<const MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::RandomAccess> > x_rnd_view_type;
typedef Kokkos::View<const MMD_float**,Kokkos::MemoryTraits<Kokkos::RandomAccess> > float_2d_rnd_view_type;
typedef Kokkos::View<const MMD_float*,Kokkos::MemoryTraits<Kokkos::RandomAccess> > float_1d_rnd_view_type;
typedef Kokkos::View<const MMD_int*,Kokkos::MemoryTraits<Kokkos::RandomAccess> > int_1d_rnd_view_type;
typedef Kokkos::View<const MMD_int**,Kokkos::MemoryTraits<Kokkos::RandomAccess> > int_2d_rnd_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_um_view_type;
typedef Kokkos::View<MMD_int*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > int_1d_um_view_type;
typedef Kokkos::View<const MMD_int*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > int_1d_const_um_view_type;

typedef typename x_view_type::HostMirror x_host_view_type;
typedef typename float_1d_view_type::HostMirror float_1d_host_view_type;
typedef typename float_2d_view_type::HostMirror float_2d_host_view_type;
typedef typename int_1d_view_type::HostMirror int_1d_host_view_type;
typedef typename int_2d_view_type::HostMirror int_2d_host_view_type;
typedef typename int_view_type::HostMirror int_host_view_type;

typedef typename Kokkos::DefaultExecutionSpace::scratch_memory_space SharedSpace;
typedef Kokkos::View<float*[3], Kokkos::LayoutLeft, SharedSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > neighbor_pos_shared_type;
typedef Kokkos::View<int*, SharedSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > int_1d_shared_type;

typedef Kokkos::View<int**,Kokkos::LayoutLeft,SharedSpace,Kokkos::MemoryUnmanaged> t_shared_2d_int;
typedef Kokkos::View<float**[3],Kokkos::LayoutLeft,SharedSpace,Kokkos::MemoryUnmanaged> t_shared_pos;

#ifdef KOKKOS_ENABLE_CUDA
typedef Kokkos::View<Kokkos::View<int*>* , Kokkos::CudaUVMSpace> t_neighlist_vov;
#else
typedef Kokkos::View<Kokkos::View<int*>*> t_neighlist_vov;
#endif

struct eng_virial_type {
  MMD_float eng;
  MMD_float virial;
  KOKKOS_INLINE_FUNCTION
  eng_virial_type() {eng = 0.0; virial = 0.0;}

  KOKKOS_INLINE_FUNCTION
  eng_virial_type& operator += (const eng_virial_type& src) {
    eng+=src.eng;
    virial+=src.virial;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  void operator += (const volatile eng_virial_type& src) volatile {
    eng+=src.eng;
    virial+=src.virial;
  }
};



#endif
