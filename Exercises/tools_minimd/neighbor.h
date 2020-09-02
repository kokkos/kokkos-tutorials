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

#ifndef NEIGHBOR_H
#define NEIGHBOR_H

#include "atom.h"
#include "timer.h"

class Neighbor
{
  public:

    typedef int value_type;

    struct TagNeighborBinning {};
    template<int HALF_NEIGH,bool STACK_ARRAYS>
    struct TagNeighborBuild {};
    
    template<int HALF_NEIGH,bool STACK_ARRAYS>
    struct TagNeighborBuildCount {};
    template<int HALF_NEIGH,bool STACK_ARRAYS>
    struct TagNeighborBuildFill {};


    int every;                       // re-neighbor every this often
    int nbinx, nbiny, nbinz;         // # of global bins
    MMD_float cutneigh;              // neighbor cutoff
    float_1d_view_type cutneighsq;   // neighbor cutoff squared
    MMD_float cutneighsq_stack[MAX_STACK_TYPES*MAX_STACK_TYPES];
    int ncalls;                      // # of times build has been called
    int max_totalneigh;              // largest # of neighbors ever stored

    int_1d_view_type numneigh;                   // # of neighbors for each atom
    int_2d_view_type neighbors;                  // array of neighbors of each atom
    t_neighlist_vov neighbors_vov;
    int maxneighs;				   // max number of neighbors per atom
    int halfneigh;
    int team_neigh_build;

    MMD_int ghost_newton;
    int count;
    Neighbor(int ntypes_);
    ~Neighbor();
    void dealloc();
    int setup(Atom &);               // setup bins based on box and cutoff
    void build(Atom &);              // create neighbor list

    Timer* timer;

    // Atom is going to call binatoms etc for sorting
    void binatoms(Atom & atom, int count = -1);           // bin all atoms

    int_1d_view_type bincount;          // ptr to 1st atom in each bin
    int_2d_view_type bins; // ptr to next atom in each bin
    int_1d_view_type bin_has_local;
    int_1d_view_type bin_list;

    int mbins;                       // binning parameters
    int mbinx, mbiny, mbinz;
    int atoms_per_bin;
    int shared_mem_size;

    KOKKOS_INLINE_FUNCTION
    void operator() (TagNeighborBinning, const int&, int&) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNeighborBinning, const int& ibin, int& offset, const bool& final) const;

    template<int HALF_NEIGH, bool STACK_ARRAYS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNeighborBuild<HALF_NEIGH,STACK_ARRAYS> ,
                     const typename Kokkos::RangePolicy<TagNeighborBuild<HALF_NEIGH,STACK_ARRAYS> >::member_type&) const;

    template<int HALF_NEIGH, bool STACK_ARRAYS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNeighborBuildCount<HALF_NEIGH,STACK_ARRAYS> ,
                     const typename Kokkos::TeamPolicy<TagNeighborBuildCount<HALF_NEIGH,STACK_ARRAYS> >::member_type&) const;


    template<int HALF_NEIGH, bool STACK_ARRAYS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNeighborBuildFill<HALF_NEIGH,STACK_ARRAYS> ,
                     const typename Kokkos::TeamPolicy<TagNeighborBuildFill<HALF_NEIGH,STACK_ARRAYS> >::member_type&) const;

    size_t team_shmem_size( int team_size ) const {
      return shared_mem_size;
    }

  private:
    MMD_float xprd, yprd, zprd;      // box size

    int nmax;                        // max size of atom arrays in neighbor
    int ntypes;                      // number of atom types

    int nstencil;                    // # of bins in stencil
    int_1d_view_type stencil;                    // stencil list of bin offsets

    int mbinxlo, mbinylo, mbinzlo;
    int nextx, nexty, nextz;
    MMD_float binsizex, binsizey, binsizez;
    MMD_float bininvx, bininvy, bininvz;

    int resize;

    KOKKOS_INLINE_FUNCTION
    MMD_float bindist(int, int, int);   // distance between binx
    KOKKOS_INLINE_FUNCTION
    int coord2bin(MMD_float, MMD_float, MMD_float) const;   // mapping atom coord to a bin

    x_rnd_view_type x;
    int_1d_rnd_view_type type;
    int nlocal;
    int_1d_view_type new_maxneighs;
    int_1d_host_view_type h_new_maxneighs;
};

#endif
