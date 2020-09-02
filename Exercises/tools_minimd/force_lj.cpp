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

#include "stdio.h"
#include "math.h"
#include "force_lj.h"

#ifndef VECTORLENGTH
#define VECTORLENGTH 4
#endif

ForceLJ::ForceLJ(int ntypes_)
{
  cutforce = 0.0;
  use_oldcompute = 0;
  reneigh = 1;
  style = FORCELJ;
  ntypes = ntypes_;

  float_1d_view_type d_cut("ForceLJ::cutforcesq",ntypes*ntypes);
  float_1d_host_view_type h_cut = Kokkos::create_mirror_view(d_cut);
  cutforcesq = d_cut;

  float_1d_view_type d_epsilon("ForceLJ::epsilon",ntypes*ntypes);
  float_1d_host_view_type h_epsilon = Kokkos::create_mirror_view(d_epsilon);
  epsilon = d_epsilon;

  float_1d_view_type d_sigma6("ForceLJ::sigma6",ntypes*ntypes);
  float_1d_host_view_type h_sigma6 = Kokkos::create_mirror_view(d_sigma6);
  sigma6 = d_sigma6;

  float_1d_view_type d_sigma("ForceLJ::sigma",ntypes*ntypes);
  float_1d_host_view_type h_sigma = Kokkos::create_mirror_view(d_sigma);
  sigma = d_sigma;

  for(int i = 0; i<ntypes*ntypes; i++) {
    h_cut[i] = 0.0;
    h_epsilon[i] = 1.0;
    h_sigma6[i] = 1.0;
    h_sigma[i] = 1.0;
    if(i<MAX_STACK_TYPES*MAX_STACK_TYPES) {
      epsilon_s[i] = 1.0;
      sigma6_s[i] = 1.0;
    }
  }

  Kokkos::deep_copy(d_cut,h_cut);
  Kokkos::deep_copy(d_epsilon,h_epsilon);
  Kokkos::deep_copy(d_sigma6,h_sigma6);
  Kokkos::deep_copy(d_sigma,h_sigma);

  nthreads = Kokkos::HostSpace::execution_space().concurrency();
}

ForceLJ::~ForceLJ() {}

void ForceLJ::setup()
{
  float_1d_view_type d_cut("ForceLJ::cutforcesq",ntypes*ntypes);
  float_1d_host_view_type h_cut = Kokkos::create_mirror_view(d_cut);
  cutforcesq = d_cut;

  for(int i = 0; i<ntypes*ntypes; i++) {
    h_cut[i] = cutforce * cutforce;
    if(i<MAX_STACK_TYPES*MAX_STACK_TYPES)
      cutforcesq_s[i] = cutforce * cutforce;
  }

  Kokkos::deep_copy(d_cut,h_cut);
}


void ForceLJ::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  eng_vdwl = 0;
  virial = 0;

#ifdef KOKKOS_HAVE_CUDA
  const int host_device = 0;
#else
  const int host_device = 1;
#endif

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;

  x = atom.x;
  f_a = atom.f;
  f = atom.f;
  type = atom.type;

  neighbors = neighbor.neighbors;
  neighbors_vov = neighbor.neighbors_vov;
  numneigh = neighbor.numneigh;

  // clear force on own and ghost atoms

  Kokkos::deep_copy(f,0.0);

  /* switch to correct compute */

  if(evflag) {
    if(use_oldcompute && host_device)
      return compute_original<1>(atom, neighbor, me);

    if(neighbor.halfneigh) {
      if(neighbor.ghost_newton) {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<1, 1>(atom, neighbor, me);
        else
          return compute_halfneigh<1, 1>(atom, neighbor, me);
      } else {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<1, 0>(atom, neighbor, me);
        else
          return compute_halfneigh<1, 0>(atom, neighbor, me);
      }
    } else return compute_fullneigh<1>(atom, neighbor, me);
  } else {
    if(use_oldcompute)
      return compute_original<0>(atom, neighbor, me);

    if(neighbor.halfneigh) {
      if(neighbor.ghost_newton) {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<0, 1>(atom, neighbor, me);
        else
          return compute_halfneigh<0, 1>(atom, neighbor, me);
      } else {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<0, 0>(atom, neighbor, me);
        else
          return compute_halfneigh<0, 0>(atom, neighbor, me);
      }
    } else return compute_fullneigh<0>(atom, neighbor, me);

  }
}

//original version of force compute in miniMD
//  -MPI only
//  -not vectorizable
template<int EVFLAG>
void ForceLJ::compute_original(Atom &atom, Neighbor &neighbor, int me)
{
  eng_vdwl = 0;
  virial = 0;

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  for(int i = 0; i < nlocal; i++) {
    const int jnum = numneigh[i];
    const MMD_float xtmp = x(i,0);
    const MMD_float ytmp = x(i,1);
    const MMD_float ztmp = x(i,2);
    const int type_i = type[i];

    for(int k = 0; k < jnum; k++) {
      const int j = neighbors(i,k);
      const MMD_float delx = xtmp - x(j,0);
      const MMD_float dely = ytmp - x(j,1);
      const MMD_float delz = ztmp - x(j,2);
      int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq(type_ij)) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6(type_ij);
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon(type_ij);
        f(i,0) += delx * force;
        f(i,1) += dely * force;
        f(i,2) += delz * force;
        f(j,0) -= delx * force;
        f(j,1) -= dely * force;
        f(j,2) -= delz * force;

        if(EVFLAG) {
          eng_vdwl += (4.0 * sr6 * (sr6 - 1.0)) * epsilon(type_ij);
          virial += (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }
  }
}


//Not Thread-safe variant of force kernel using half-neighborlists
template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ::compute_halfneigh(Atom &atom, Neighbor &neighbor, int me)
{

  // loop over all neighbors of my atoms
  // store force on both atoms i and j
  MMD_float t_energy = 0;
  MMD_float t_virial = 0;

  for(int i = 0; i < nlocal; i++) {
    const int numneighs = numneigh[i];
    const MMD_float xtmp = x(i,0);
    const MMD_float ytmp = x(i,1);
    const MMD_float ztmp = x(i,2);
    const int type_i = type[i];

    MMD_float fix = 0.0;
    MMD_float fiy = 0.0;
    MMD_float fiz = 0.0;

#ifdef USE_SIMD
    #pragma simd reduction (+: fix,fiy,fiz)
#endif
    for(int k = 0; k < numneighs; k++) {
      const int j = neighbors(i,k);

      const MMD_float delx = xtmp - x(j,0);
      const MMD_float dely = ytmp - x(j,1);
      const MMD_float delz = ztmp - x(j,2);
      const int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;
      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq(type_ij)) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6(type_ij);
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon(type_ij);

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal) {
          f(j,0) -= delx * force;
          f(j,1) -= dely * force;
          f(j,2) -= delz * force;
        }

        if(EVFLAG) {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
          t_energy += scale * (4.0 * sr6 * (sr6 - 1.0)) * epsilon(type_ij);
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }

      }
    }

    f(i,0) += fix;
    f(i,1) += fiy;
    f(i,2) += fiz;

  }

  eng_vdwl += t_energy;
  virial += t_virial;

}

//Thread-safe variant of force kernel using half-neighborlists with atomics
template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ::compute_halfneigh_threaded(Atom &atom, Neighbor &neighbor, int me)
{
  eng_virial_type t_eng_virial;

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  if(ntypes>MAX_STACK_TYPES) {
    if(EVFLAG)
      Kokkos::parallel_reduce("ForceLJ::compute_half_ev",Kokkos::RangePolicy<TagComputeHalfNeighThread<1,GHOST_NEWTON,0> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for("ForceLJ::compute_half",Kokkos::RangePolicy<TagComputeHalfNeighThread<0,GHOST_NEWTON,0> >(0,nlocal), *this );
  } else {
    if(EVFLAG)
      Kokkos::parallel_reduce("ForceLJ::compute_half_ev",Kokkos::RangePolicy<TagComputeHalfNeighThread<1,GHOST_NEWTON,1> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for("ForceLJ::compute_half",Kokkos::RangePolicy<TagComputeHalfNeighThread<0,GHOST_NEWTON,1> >(0,nlocal), *this );
  }
  eng_vdwl += t_eng_virial.eng;
  virial += t_eng_virial.virial;
}

//Thread-safe variant of force kernel using full-neighborlists
//   -trades more calculation for no atomics
//   -compared to halgneigh_threads:
//        2x reads, 0x writes (reads+writes the same as with half)
//        2x flops
template<int EVFLAG>
void ForceLJ::compute_fullneigh(Atom &atom, Neighbor &neighbor, int me)
{
  eng_virial_type t_eng_virial;

  // loop over all neighbors of my atoms
  // store force on atom i

  if(ntypes>MAX_STACK_TYPES) {
    if(EVFLAG)
      Kokkos::parallel_reduce("ForceLJ::compute_full_ev",Kokkos::RangePolicy<TagComputeFullNeigh<1,0> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for("ForceLJ::compute_full",Kokkos::RangePolicy<TagComputeFullNeigh<0,0> >(0,nlocal), *this );
  } else {
    if(EVFLAG)
      Kokkos::parallel_reduce("ForceLJ::compute_full_ev",Kokkos::RangePolicy<TagComputeFullNeigh<1,1> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for("ForceLJ::compute_full",Kokkos::RangePolicy<TagComputeFullNeigh<0,1> >(0,nlocal), *this );
  }
  t_eng_virial.eng *= 4.0;
  t_eng_virial.virial *= 0.5;

  eng_vdwl += t_eng_virial.eng;
  virial += t_eng_virial.virial;
}

template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> , const int& i) const {
  eng_virial_type dummy;
  this->operator()(TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> (), i, dummy);
}

template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> , const int& i, eng_virial_type& eng_virial) const {

  const int numneighs = numneigh[i];

  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];

  int_1d_view_type neighs_i = neighbors_vov(i);
  MMD_float fix = 0.0;
  MMD_float fiy = 0.0;
  MMD_float fiz = 0.0;

  for(int k = 0; k < numneighs; k++) {
    const MMD_int j = neighs_i(k);
    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;
    const int type_ij = type_i*ntypes+type_j;

    if(rsq < (STACK_PARAMS?cutforcesq_s[type_ij]:cutforcesq(type_ij))) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2 * (STACK_PARAMS?sigma6_s[type_ij]:sigma6(type_ij));
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));

      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      if(GHOST_NEWTON || j < nlocal) {
        f_a(j,0) -= delx * force;
        f_a(j,1) -= dely * force;
        f_a(j,2) -= delz * force;
      }

      if(EVFLAG) {
        const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
        eng_virial.eng += scale * 4.0 * sr6 * (sr6 - 1.0) * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
        eng_virial.virial += scale * (delx * delx + dely * dely + delz * delz) * force;
      }
    }
  }

  f_a(i,0) += fix;
  f_a(i,1) += fiy;
  f_a(i,2) += fiz;
}

template<int EVFLAG, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeFullNeigh<EVFLAG,STACK_PARAMS> , const int& i) const {
  eng_virial_type dummy;
  this->operator()(TagComputeFullNeigh<EVFLAG,STACK_PARAMS> (), i, dummy);
}

template<int EVFLAG, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeFullNeigh<EVFLAG,STACK_PARAMS> , const int& i, eng_virial_type& eng_virial) const {

  const int numneighs = numneigh[i];

  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];
  int_1d_view_type neighs_i = neighbors_vov(i);

  MMD_float fix = 0;
  MMD_float fiy = 0;
  MMD_float fiz = 0;

  //pragma simd forces vectorization (ignoring the performance objections of the compiler)
  //also give hint to use certain vectorlength for MIC, Sandy Bridge and WESTMERE this should be be 8 here
  //give hint to compiler that fix, fiy and fiz are used for reduction only

#ifdef USE_SIMD
  #pragma simd reduction (+: fix,fiy,fiz,eng_virial)
#endif
  for(int k = 0; k < numneighs; k++) {
    const MMD_int j = neighs_i(k);
    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;
    int type_ij = type_i*ntypes+type_j;

    if(rsq < (STACK_PARAMS?cutforcesq_s[type_ij]:cutforcesq(type_ij))) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2 * (STACK_PARAMS?sigma6_s[type_ij]:sigma6(type_ij));
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      if(EVFLAG) {
        eng_virial.eng += sr6 * (sr6 - 1.0) * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
        eng_virial.virial += (delx * delx + dely * dely + delz * delz) * force;
      }
    }

  }

  f(i,0) += fix;
  f(i,1) += fiy;
  f(i,2) += fiz;
}

