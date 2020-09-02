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
#include "stdlib.h"

#include "neighbor.h"

#define FACTOR 0.999
#define SMALL 1.0e-6


Neighbor::Neighbor(int ntypes_)
{
  ncalls = 0;
  ntypes = ntypes_;
  max_totalneigh = 0;
  maxneighs = 100;
  nmax = 0;
  atoms_per_bin = 8;
  halfneigh = 0;
  ghost_newton = 1;

  cutneighsq = float_1d_view_type("Neighbor::cutneighsq",ntypes*ntypes);
  new_maxneighs = int_1d_view_type("Neighbor::new_maxneighs",1);
  h_new_maxneighs = Kokkos::create_mirror_view(new_maxneighs);
  team_neigh_build = 1;

  shared_mem_size = 0;
}

Neighbor::~Neighbor()
{
}

void Neighbor::dealloc() {
  for(int i=0; i<nmax; i++)
    neighbors_vov(i) = int_1d_view_type();
  neighbors_vov = t_neighlist_vov();
}
/* binned neighbor list construction with full Newton's 3rd law
   every pair stored exactly once by some processor
   each owned atom i checks its own bin and other bins in Newton stencil */

void Neighbor::build(Atom &atom)
{
  ncalls++;
  nlocal = atom.nlocal;
  const int nall = atom.nlocal + atom.nghost;
  /* extend atom arrays if necessary */

  if(nall > nmax) {
    nmax = nall;

    numneigh = int_1d_view_type("Neighbor::numneigh",nmax);
    neighbors = int_2d_view_type("Neighbor::neighbors",nmax , maxneighs);
    neighbors_vov = t_neighlist_vov("Neighbor::neighbors_vov_outer",nmax);
  }

  /* bin local & ghost atoms */

  binatoms(atom);
  count = 0;

  x = atom.x;
  type = atom.type;
  ntypes = atom.ntypes;

  resize = 1;


  /* repeat calculations if running out of memory */
    Kokkos::deep_copy(new_maxneighs , maxneighs);
    resize = 0;

    /* loop over each atom, storing neighbors */
    /* flat parallelism particle based, and team based with a bin per team */
    if(ntypes<MAX_STACK_TYPES) {
        int team_size = team_neigh_build;
        int vector_length = 32;
        while(vector_length>atoms_per_bin) vector_length/=2;
        shared_mem_size = (2*team_size +2*nextx) * atoms_per_bin * (3*sizeof(float) + 2 * sizeof(int));
        if(halfneigh)
          Kokkos::parallel_for("Neighbor::Count",Kokkos::TeamPolicy<TagNeighborBuildCount<1,1> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        else
          Kokkos::parallel_for("Neighbor::Count",Kokkos::TeamPolicy<TagNeighborBuildCount<0,1> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        shared_mem_size = 0;
    } else {
        int team_size = team_neigh_build;
        int vector_length = 32;
        while(vector_length>atoms_per_bin) vector_length/=2;
        shared_mem_size = (2*team_size +2*nextx) * atoms_per_bin * (3*sizeof(float) + 2 * sizeof(int));
        if(halfneigh)
          Kokkos::parallel_for("Neighbor::Count",Kokkos::TeamPolicy<TagNeighborBuildCount<1,0> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        else
          Kokkos::parallel_for("Neighbor::Count",Kokkos::TeamPolicy<TagNeighborBuildCount<0,0> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        shared_mem_size = 0;
    }

    Kokkos::deep_copy(h_new_maxneighs,new_maxneighs);
    if(h_new_maxneighs(0) > maxneighs) {
      resize = 1;
      maxneighs = h_new_maxneighs(0) * 1.2;
      neighbors = int_2d_view_type("Neighbor::neighbors", nmax , maxneighs);
    }
    int_1d_host_view_type h_numneigh = Kokkos::create_mirror_view(numneigh);
    Kokkos::deep_copy(h_numneigh,numneigh);
    for(int i=0; i<nmax; i++) {
      neighbors_vov(i) = Kokkos::View<int*>("Neighbors::neighbors_vov",h_numneigh(i));
    }
    if(ntypes<MAX_STACK_TYPES) {
        int team_size = team_neigh_build;
        int vector_length = 32;
        while(vector_length>atoms_per_bin) vector_length/=2;
        shared_mem_size = (2*team_size +2*nextx) * atoms_per_bin * (3*sizeof(float) + 2 * sizeof(int));
        if(halfneigh)
          Kokkos::parallel_for("Neighbor::Fill",Kokkos::TeamPolicy<TagNeighborBuildFill<1,1> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        else
          Kokkos::parallel_for("Neighbor::Fill",Kokkos::TeamPolicy<TagNeighborBuildFill<0,1> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        shared_mem_size = 0;
    } else {
        int team_size = team_neigh_build;
        int vector_length = 32;
        while(vector_length>atoms_per_bin) vector_length/=2;
        shared_mem_size = (2*team_size +2*nextx) * atoms_per_bin * (3*sizeof(float) + 2 * sizeof(int));
        if(halfneigh)
          Kokkos::parallel_for("Neighbor::Fill",Kokkos::TeamPolicy<TagNeighborBuildFill<1,0> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        else
          Kokkos::parallel_for("Neighbor::Fill",Kokkos::TeamPolicy<TagNeighborBuildFill<0,0> >((mbinx-2*nextx)*(mbiny-2*nexty)*(mbinz-2*nextz)/team_size,team_size,vector_length), *this);
        shared_mem_size = 0;
    }

}

template<int HALF_NEIGH,bool STACK_ARRAYS>
KOKKOS_INLINE_FUNCTION
void Neighbor::operator() (TagNeighborBuild<HALF_NEIGH,STACK_ARRAYS> , const typename Kokkos::RangePolicy<TagNeighborBuild<HALF_NEIGH,STACK_ARRAYS> >::member_type& i) const {
  int n = 0;

  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);

  const int type_i = type[i];

  const int ibin = coord2bin(xtmp, ytmp, ztmp);

  for(int k = 0; k < nstencil; k++) {
    const int jbin = ibin + stencil[k];

//    int* loc_bin = &bins(jbin,0);

    if(ibin == jbin)
      for(int m = 0; m < bincount[jbin]; m++) {
        const int j = bins(jbin,m);

        //for same bin as atom i skip j if i==j and skip atoms "below and to the left" if using halfneighborlists
        if(((j == i) || (HALF_NEIGH && !ghost_newton && (j < i)) ||
            (HALF_NEIGH && ghost_newton && ((j < i) || ((j >= nlocal) &&
                                           ((x(j,2) < ztmp) || (x(j,2) == ztmp && x(j,1) < ytmp) ||
                                            (x(j,2) == ztmp && x(j,1)  == ytmp && x(j,0) < xtmp))))))) continue;

        const MMD_float delx = xtmp - x(j,0);
        const MMD_float dely = ytmp - x(j,1);
        const MMD_float delz = ztmp - x(j,2);
        const int type_j = type[j];
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        if(rsq <= (STACK_ARRAYS?cutneighsq_stack[type_i*ntypes+type_j]:cutneighsq[type_i*ntypes+type_j])) {
          if(n<maxneighs) neighbors(i,n) = j;
          n++;
        }
      }
    else {
      for(int m = 0; m < bincount[jbin]; m++) {
        const int j = bins(jbin,m);

        if(halfneigh && !ghost_newton && (j < i)) continue;

        const MMD_float delx = xtmp - x(j,0);
        const MMD_float dely = ytmp - x(j,1);
        const MMD_float delz = ztmp - x(j,2);
        const int type_j = type[j];
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        if(rsq <= (STACK_ARRAYS?cutneighsq_stack[type_i*ntypes+type_j]:cutneighsq[type_i*ntypes+type_j])) {
          if(n<maxneighs) neighbors(i,n) = j;
          n++;
        }
      }
    }
  }

  numneigh[i] = n;

  if(n >= maxneighs) {
    if(n >= new_maxneighs(0)) new_maxneighs(0) = n;
  }
}

template<int HALF_NEIGH, bool STACK_ARRAYS>
KOKKOS_INLINE_FUNCTION
void Neighbor::operator() (TagNeighborBuildCount<HALF_NEIGH,STACK_ARRAYS> , const typename Kokkos::TeamPolicy<TagNeighborBuildCount<HALF_NEIGH,STACK_ARRAYS> >::member_type& team_member) const {

  const int atoms_per_bin = bins.extent(1);

  // Each thread gets one bin
  const int binoffset = team_member.league_rank()*team_member.team_size() + team_member.team_rank();

  // Calculate the 3D index of the bin of this thread. Threads in a team own neighboring bins in X directions
  const int binx = binoffset%(mbinx-2*nextx) + nextx;
  const int biny = (binoffset/(mbinx-2*nextx))%(mbiny-2*nexty) + nexty;
  const int binz = (binoffset/((mbinx-2*nextx)*(mbiny-2*nexty)))%(mbinz-2*nextz) + nextz;
  const int ibin = binz*mbiny*mbinx + biny*mbinx + binx;

  // Local bin index within the team (just the team_rank();
  const int MY_BIN = team_member.team_rank();

  // Create shared allocations for the owned bins
  t_shared_2d_int this_type(team_member.team_shmem(),team_member.team_size(),atoms_per_bin);
  t_shared_2d_int this_id(team_member.team_shmem(),team_member.team_size(),atoms_per_bin);
  t_shared_pos    this_x(team_member.team_shmem(),atoms_per_bin,team_member.team_size());
  // Create shared allocations for one x row in the common neighbor stencil of the owned bins of the team
  t_shared_2d_int other_type(team_member.team_shmem(),team_member.team_size()+2*nextx,atoms_per_bin);
  t_shared_2d_int other_id(team_member.team_shmem(),team_member.team_size()+2*nextx,atoms_per_bin);
  t_shared_pos    other_x(team_member.team_shmem(),atoms_per_bin,team_member.team_size()+2*nextx);

  // Get the count of atoms in the owned bin
  int bincount_current = ibin >=bincount.extent(0)?0:bincount[ibin];

  // Load atoms in the owned bin. Each Thread loads one bin
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,bincount_current), [&] (const int& ii) {
    const int i = bins(ibin,ii);
    this_x(ii,MY_BIN, 0) = x(i, 0);
    this_x(ii,MY_BIN, 1) = x(i, 1);
    this_x(ii,MY_BIN, 2) = x(i, 2);
    this_type(MY_BIN,ii) = type(i);
    this_id(MY_BIN,ii) = i;
  });

  // No barrier necessary since only vector loops follow
  //team_member.team_barrier();

  // Calculate interactions with atoms in owned bin. Split work over vector lanes
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,bincount_current), [&] (const int& ii) {
    const MMD_float xtmp = this_x(ii,MY_BIN,0);
    const MMD_float ytmp = this_x(ii,MY_BIN,1);
    const MMD_float ztmp = this_x(ii,MY_BIN,2);
    const int itype = this_type(MY_BIN,ii);
    const int i = this_id(MY_BIN,ii);

    // Only calculate neighborlist if the atom is NOT a ghost atom in the MPI domain
    if(i<nlocal) {

    // The local neighbor count for this atom
    int n = 0;

    // Get the reference to the neighbor list of atom i
    // (a very slimmed down variant of subview which is faster to generate and only has two members)
    //const AtomNeighbors neighbors_i = neigh_list.get_neighbors(i);

    // Loop over atoms in owned bin for i,j check
    #pragma unroll 4
    for(int m = 0; m < bincount_current; m++) {

      const int j = this_id(MY_BIN,m);
      const int jtype = this_type(MY_BIN,m);

      // Exclude invalid atoms for HALF_NEIGH / GhostNewton etc.
      if((j == i) ||
         (HALF_NEIGH && !ghost_newton && (j < i))  ||
         (HALF_NEIGH && ghost_newton &&
            ((j < i) ||
            ((j >= nlocal) && ((x(j, 2) < ztmp) || (x(j, 2) == ztmp && x(j, 1) < ytmp) ||
              (x(j, 2) == ztmp && x(j, 1)  == ytmp && x(j, 0) < xtmp)))))
        ) continue;

      // Calculate i-j distance
      const MMD_float delx = xtmp - this_x(m,MY_BIN,0);
      const MMD_float dely = ytmp - this_x(m,MY_BIN,1);
      const MMD_float delz = ztmp - this_x(m,MY_BIN,2);
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      // If i-j distance smaller than cutoff add it to the neighbor list
      // Only add the atom if neighborlist is large enough, but always keep counting
      if(rsq <= (STACK_ARRAYS?cutneighsq_stack[itype*ntypes+jtype]:cutneighsq[itype*ntypes+jtype])) {
        n++;
      }

    }
    // Store the number of neighbors
    numneigh(i) = n;
    }
  });
  const int zstart = (HALF_NEIGH && ghost_newton)?0:-nextz;
  // Loop over stencil in Y and Z direction
  for (int zz = zstart; zz <= nextz; zz++)
    for (int yy = -nexty; yy <= nexty; yy++) {

      // Load the x-row of the neighbor stencil of the owned bins into shared memory
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,team_member.team_size()+2*nextx), [&] (const int xx) {
        const int team_start_bin = ibin-team_member.team_rank();
        const int jbin = team_start_bin + zz*mbiny*mbinx + yy*mbinx + xx - nextx;
        const int j_bincount_current = (jbin>=bincount.extent(0)?0:bincount[jbin]);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,j_bincount_current), [&] (const int& jj) {
          const int j = bins(jbin,jj);
          other_x(jj,xx,0) = x(j, 0);
          other_x(jj,xx,1) = x(j, 1);
          other_x(jj,xx,2) = x(j, 2);
          other_type(xx,jj) = type(j);
          other_id(xx,jj) = j;
        });
      });
      // Wait for all threads to finish filling shared memory with current x row of neighbor bins
      team_member.team_barrier();

    // Vector Loop over atoms in the owned bin of this thread
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,bincount_current), [&] (const int& ii) {
      const MMD_float xtmp = this_x(ii,MY_BIN,0);
      const MMD_float ytmp = this_x(ii,MY_BIN,1);
      const MMD_float ztmp = this_x(ii,MY_BIN,2);
      const int itype = this_type(MY_BIN,ii);
      const int i = this_id(MY_BIN,ii);

      // Only calculate neighborlist if the atom is NOT a ghost atom in the MPI domain
      if(i<nlocal) {

        // Load the neighbor count from previous iteration over the stencil
        int n = numneigh(i);

        // Loop over neighbor bins in X directions which are in shared memory
        for (int xx = 0; xx <= 2*nextx; xx++) {
          // Exclude the central bin in the stencil which is the owned bin and was done previously
          if(zz==0 && yy==0 && xx==nextx) continue;

          // Exclude bins not part of half neigh stencil
          if(HALF_NEIGH && ghost_newton && (zz == 0 && ( yy < 0 || (yy == 0 && xx < nextx)))) continue;
          // What is the current neighbor bin of this threads owned bin
          const int jbin = team_member.team_rank() + xx;
          // Get the atom count in that bin
          const int j_bincount_current = jbin>=bincount.extent(0)?0:bincount[ibin+zz*mbiny*mbinx + yy*mbinx + xx - nextx];

          // Loop over the neighbor bin
          #pragma unroll 8
          for(int m = 0; m < j_bincount_current; m++) {
            const int j = other_id(jbin,m);
            const int jtype = other_type(jbin,m);

            if(HALF_NEIGH && !ghost_newton && (j < i)) continue;

            // Calculate i-j distance
            const MMD_float delx = xtmp - other_x(m,jbin,0);
            const MMD_float dely = ytmp - other_x(m,jbin,1);
            const MMD_float delz = ztmp - other_x(m,jbin,2);
            const MMD_float rsq = delx * delx + dely * dely + delz * delz;

            // If i-j distance smaller than cutoff add it to the neighbor list
            // Only add the atom if neighborlist is large enough, but always keep counting
            if(rsq <= (STACK_ARRAYS?cutneighsq_stack[itype*ntypes+jtype]:cutneighsq[itype*ntypes+jtype])) {
              n++;
            }
          }

        }
        // Store this atoms count
        numneigh(i) = n;

        // Set regrow flag if neccessary
        if(n >= maxneighs) {
          if(n >= new_maxneighs(0)) new_maxneighs(0) = n;
        }        
      }
    });

    // Wait for all threads to finish with current x row in shared memory
    team_member.team_barrier();

  }
}


template<int HALF_NEIGH, bool STACK_ARRAYS>
KOKKOS_INLINE_FUNCTION
void Neighbor::operator() (TagNeighborBuildFill<HALF_NEIGH,STACK_ARRAYS> , const typename Kokkos::TeamPolicy<TagNeighborBuildFill<HALF_NEIGH,STACK_ARRAYS> >::member_type& team_member) const {

  const int atoms_per_bin = bins.extent(1);

  // Each thread gets one bin
  const int binoffset = team_member.league_rank()*team_member.team_size() + team_member.team_rank();

  // Calculate the 3D index of the bin of this thread. Threads in a team own neighboring bins in X directions
  const int binx = binoffset%(mbinx-2*nextx) + nextx;
  const int biny = (binoffset/(mbinx-2*nextx))%(mbiny-2*nexty) + nexty;
  const int binz = (binoffset/((mbinx-2*nextx)*(mbiny-2*nexty)))%(mbinz-2*nextz) + nextz;
  const int ibin = binz*mbiny*mbinx + biny*mbinx + binx;

  // Local bin index within the team (just the team_rank();
  const int MY_BIN = team_member.team_rank();

  // Create shared allocations for the owned bins
  t_shared_2d_int this_type(team_member.team_shmem(),team_member.team_size(),atoms_per_bin);
  t_shared_2d_int this_id(team_member.team_shmem(),team_member.team_size(),atoms_per_bin);
  t_shared_pos    this_x(team_member.team_shmem(),atoms_per_bin,team_member.team_size());
  // Create shared allocations for one x row in the common neighbor stencil of the owned bins of the team
  t_shared_2d_int other_type(team_member.team_shmem(),team_member.team_size()+2*nextx,atoms_per_bin);
  t_shared_2d_int other_id(team_member.team_shmem(),team_member.team_size()+2*nextx,atoms_per_bin);
  t_shared_pos    other_x(team_member.team_shmem(),atoms_per_bin,team_member.team_size()+2*nextx);

  // Get the count of atoms in the owned bin
  int bincount_current = ibin >=bincount.extent(0)?0:bincount[ibin];

  // Load atoms in the owned bin. Each Thread loads one bin
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,bincount_current), [&] (const int& ii) {
    const int i = bins(ibin,ii);
    this_x(ii,MY_BIN, 0) = x(i, 0);
    this_x(ii,MY_BIN, 1) = x(i, 1);
    this_x(ii,MY_BIN, 2) = x(i, 2);
    this_type(MY_BIN,ii) = type(i);
    this_id(MY_BIN,ii) = i;
  });

  // No barrier necessary since only vector loops follow
  //team_member.team_barrier();

  // Calculate interactions with atoms in owned bin. Split work over vector lanes
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,bincount_current), [&] (const int& ii) {
    const MMD_float xtmp = this_x(ii,MY_BIN,0);
    const MMD_float ytmp = this_x(ii,MY_BIN,1);
    const MMD_float ztmp = this_x(ii,MY_BIN,2);
    const int itype = this_type(MY_BIN,ii);
    const int i = this_id(MY_BIN,ii);

    // Only calculate neighborlist if the atom is NOT a ghost atom in the MPI domain
    if(i<nlocal) {
    int_1d_view_type neighs_i = neighbors_vov(i);

    // The local neighbor count for this atom
    int n = 0;

    // Get the reference to the neighbor list of atom i
    // (a very slimmed down variant of subview which is faster to generate and only has two members)
    //const AtomNeighbors neighbors_i = neigh_list.get_neighbors(i);

    // Loop over atoms in owned bin for i,j check
    #pragma unroll 4
    for(int m = 0; m < bincount_current; m++) {

      const int j = this_id(MY_BIN,m);
      const int jtype = this_type(MY_BIN,m);

      // Exclude invalid atoms for HALF_NEIGH / GhostNewton etc.
      if((j == i) ||
         (HALF_NEIGH && !ghost_newton && (j < i))  ||
         (HALF_NEIGH && ghost_newton &&
            ((j < i) ||
            ((j >= nlocal) && ((x(j, 2) < ztmp) || (x(j, 2) == ztmp && x(j, 1) < ytmp) ||
              (x(j, 2) == ztmp && x(j, 1)  == ytmp && x(j, 0) < xtmp)))))
        ) continue;

      // Calculate i-j distance
      const MMD_float delx = xtmp - this_x(m,MY_BIN,0);
      const MMD_float dely = ytmp - this_x(m,MY_BIN,1);
      const MMD_float delz = ztmp - this_x(m,MY_BIN,2);
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      // If i-j distance smaller than cutoff add it to the neighbor list
      // Only add the atom if neighborlist is large enough, but always keep counting
      if(rsq <= (STACK_ARRAYS?cutneighsq_stack[itype*ntypes+jtype]:cutneighsq[itype*ntypes+jtype])) {
        neighs_i(n) = j;
        n++;
      }

    }
    // Store the number of neighbors
    numneigh(i) = n;
    }
  });
  const int zstart = (HALF_NEIGH && ghost_newton)?0:-nextz;
  // Loop over stencil in Y and Z direction
  for (int zz = zstart; zz <= nextz; zz++)
    for (int yy = -nexty; yy <= nexty; yy++) {

      // Load the x-row of the neighbor stencil of the owned bins into shared memory
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,team_member.team_size()+2*nextx), [&] (const int xx) {
        const int team_start_bin = ibin-team_member.team_rank();
        const int jbin = team_start_bin + zz*mbiny*mbinx + yy*mbinx + xx - nextx;
        const int j_bincount_current = (jbin>=bincount.extent(0)?0:bincount[jbin]);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,j_bincount_current), [&] (const int& jj) {
          const int j = bins(jbin,jj);
          other_x(jj,xx,0) = x(j, 0);
          other_x(jj,xx,1) = x(j, 1);
          other_x(jj,xx,2) = x(j, 2);
          other_type(xx,jj) = type(j);
          other_id(xx,jj) = j;
        });
      });
      // Wait for all threads to finish filling shared memory with current x row of neighbor bins
      team_member.team_barrier();

    // Vector Loop over atoms in the owned bin of this thread
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member,bincount_current), [&] (const int& ii) {
      const MMD_float xtmp = this_x(ii,MY_BIN,0);
      const MMD_float ytmp = this_x(ii,MY_BIN,1);
      const MMD_float ztmp = this_x(ii,MY_BIN,2);
      const int itype = this_type(MY_BIN,ii);
      const int i = this_id(MY_BIN,ii);

      // Only calculate neighborlist if the atom is NOT a ghost atom in the MPI domain
      if(i<nlocal) {
        int_1d_view_type neighs_i = neighbors_vov(i);

        // Load the neighbor count from previous iteration over the stencil
        int n = numneigh(i);

        // Loop over neighbor bins in X directions which are in shared memory
        for (int xx = 0; xx <= 2*nextx; xx++) {
          // Exclude the central bin in the stencil which is the owned bin and was done previously
          if(zz==0 && yy==0 && xx==nextx) continue;

          // Exclude bins not part of half neigh stencil
          if(HALF_NEIGH && ghost_newton && (zz == 0 && ( yy < 0 || (yy == 0 && xx < nextx)))) continue;
          // What is the current neighbor bin of this threads owned bin
          const int jbin = team_member.team_rank() + xx;
          // Get the atom count in that bin
          const int j_bincount_current = jbin>=bincount.extent(0)?0:bincount[ibin+zz*mbiny*mbinx + yy*mbinx + xx - nextx];

          // Loop over the neighbor bin
          #pragma unroll 8
          for(int m = 0; m < j_bincount_current; m++) {
            const int j = other_id(jbin,m);
            const int jtype = other_type(jbin,m);

            if(HALF_NEIGH && !ghost_newton && (j < i)) continue;

            // Calculate i-j distance
            const MMD_float delx = xtmp - other_x(m,jbin,0);
            const MMD_float dely = ytmp - other_x(m,jbin,1);
            const MMD_float delz = ztmp - other_x(m,jbin,2);
            const MMD_float rsq = delx * delx + dely * dely + delz * delz;

            // If i-j distance smaller than cutoff add it to the neighbor list
            // Only add the atom if neighborlist is large enough, but always keep counting
            if(rsq <= (STACK_ARRAYS?cutneighsq_stack[itype*ntypes+jtype]:cutneighsq[itype*ntypes+jtype])) {
              neighs_i(n) = j;
              n++;
            }
          }

        }
        // Store this atoms count
        numneigh(i) = n;
      }
    });

    // Wait for all threads to finish with current x row in shared memory
    team_member.team_barrier();

  }
}

void Neighbor::binatoms(Atom &atom, int count)
{
  const int nall = count<0?atom.nlocal + atom.nghost:count;
  x = atom.x;

  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  resize = 1;


  /* repeat if running out of space */

  while(resize > 0) {
    Kokkos::fence();
    resize = 0;

    Kokkos::deep_copy(bincount,0);
    Kokkos::deep_copy(bin_has_local,0);

    Kokkos::fence();
    /* count aotms in each bin */
    Kokkos::parallel_reduce("Neighbor:binning",Kokkos::RangePolicy<TagNeighborBinning>(0,nall), *this, resize);

    if(resize) {
      atoms_per_bin *= 2;
      bins = int_2d_view_type("Neighbor::bins", mbins , atoms_per_bin);
    }
  }

  Kokkos::deep_copy(bin_list,-1);
  Kokkos::parallel_scan("Neighbor::binning_scan",Kokkos::RangePolicy<TagNeighborBinning>(0,mbins), *this);
}


KOKKOS_INLINE_FUNCTION
void Neighbor::operator() (TagNeighborBinning, const int& i, int& resize) const{
  const int ibin = coord2bin(x(i,0), x(i,1), x(i,2));

  const int ac = Kokkos::atomic_fetch_add(&bincount(ibin),1);

  if(ac < atoms_per_bin) {
    bins(ibin, ac) = i;
    if(i<nlocal) bin_has_local(ibin) = 1;
  } else resize += 1;
}

KOKKOS_INLINE_FUNCTION
void Neighbor::operator() (TagNeighborBinning, const int& ibin, int& offset, const bool& final) const{
  if(bin_has_local(ibin)) {
    if(final)
      bin_list(offset) = ibin;
    offset++;
  }
}

/* convert xyz atom coords into local bin #
   take special care to insure ghost atoms with
   coord >= prd or coord < 0.0 are put in correct bins */
KOKKOS_INLINE_FUNCTION
int Neighbor::coord2bin(MMD_float x, MMD_float y, MMD_float z) const
{
  int ix, iy, iz;

  if(x >= xprd)
    ix = (int)((x - xprd) * bininvx) + nbinx - mbinxlo;
  else if(x >= 0.0)
    ix = (int)(x * bininvx) - mbinxlo;
  else
    ix = (int)(x * bininvx) - mbinxlo - 1;

  if(y >= yprd)
    iy = (int)((y - yprd) * bininvy) + nbiny - mbinylo;
  else if(y >= 0.0)
    iy = (int)(y * bininvy) - mbinylo;
  else
    iy = (int)(y * bininvy) - mbinylo - 1;

  if(z >= zprd)
    iz = (int)((z - zprd) * bininvz) + nbinz - mbinzlo;
  else if(z >= 0.0)
    iz = (int)(z * bininvz) - mbinzlo;
  else
    iz = (int)(z * bininvz) - mbinzlo - 1;

  return (iz * mbiny * mbinx + iy * mbinx + ix + 1);
}


/*
setup neighbor binning parameters
bin numbering is global: 0 = 0.0 to binsize
                         1 = binsize to 2*binsize
                         nbin-1 = prd-binsize to binsize
                         nbin = prd to prd+binsize
                         -1 = -binsize to 0.0
coord = lowest and highest values of ghost atom coords I will have
        add in "small" for round-off safety
mbinlo = lowest global bin any of my ghost atoms could fall into
mbinhi = highest global bin any of my ghost atoms could fall into
mbin = number of bins I need in a dimension
stencil() = bin offsets in 1-d sense for stencil of surrounding bins
*/

int Neighbor::setup(Atom &atom)
{
  int i, j, k, nmax;
  MMD_float coord;
  int mbinxhi, mbinyhi, mbinzhi;

  float_1d_host_view_type h_cutneighsq = Kokkos::create_mirror_view(cutneighsq);   // neighbor cutoff squared

  for(int i = 0; i<ntypes*ntypes; i++) {
    h_cutneighsq(i) = cutneigh * cutneigh;
    if(i<MAX_STACK_TYPES*MAX_STACK_TYPES)
    cutneighsq_stack[i] = cutneigh * cutneigh;
  }

  Kokkos::deep_copy(cutneighsq,h_cutneighsq);
  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  /*
  c bins must evenly divide into box size,
  c   becoming larger than cutneigh if necessary
  c binsize = 1/2 of cutoff is near optimal

  if (flag == 0) {
    nbinx = 2.0 * xprd / cutneigh;
    nbiny = 2.0 * yprd / cutneigh;
    nbinz = 2.0 * zprd / cutneigh;
    if (nbinx == 0) nbinx = 1;
    if (nbiny == 0) nbiny = 1;
    if (nbinz == 0) nbinz = 1;
  }
  */

  int check = 1;
  while(check) {
    check = 0;
    binsizex = xprd / nbinx;
    binsizey = yprd / nbiny;
    binsizez = zprd / nbinz;
    bininvx = 1.0 / binsizex;
    bininvy = 1.0 / binsizey;
    bininvz = 1.0 / binsizez;

    coord = atom.box.xlo - cutneigh - SMALL * xprd;
    mbinxlo = static_cast<int>(coord * bininvx);

    if(coord < 0.0) mbinxlo = mbinxlo - 1;

    coord = atom.box.xhi + cutneigh + SMALL * xprd;
    mbinxhi = static_cast<int>(coord * bininvx);

    coord = atom.box.ylo - cutneigh - SMALL * yprd;
    mbinylo = static_cast<int>(coord * bininvy);

    if(coord < 0.0) mbinylo = mbinylo - 1;

    coord = atom.box.yhi + cutneigh + SMALL * yprd;
    mbinyhi = static_cast<int>(coord * bininvy);

    coord = atom.box.zlo - cutneigh - SMALL * zprd;
    mbinzlo = static_cast<int>(coord * bininvz);

    if(coord < 0.0) mbinzlo = mbinzlo - 1;

    coord = atom.box.zhi + cutneigh + SMALL * zprd;
    mbinzhi = static_cast<int>(coord * bininvz);

    /* extend bins by 1 in each direction to insure stencil coverage */

    mbinxlo = mbinxlo - 1;
    mbinxhi = mbinxhi + 1;
    mbinx = mbinxhi - mbinxlo + 1;

    mbinylo = mbinylo - 1;
    mbinyhi = mbinyhi + 1;
    mbiny = mbinyhi - mbinylo + 1;

    mbinzlo = mbinzlo - 1;
    mbinzhi = mbinzhi + 1;
    mbinz = mbinzhi - mbinzlo + 1;
    /*
    compute bin stencil of all bins whose closest corner to central bin
    is within neighbor cutoff
    for partial Newton (newton = 0),
    stencil is all surrounding bins including self
    for full Newton (newton = 1),
    stencil is bins to the "upper right" of central bin, does NOT include self
    next(xyz) = how far the stencil could possibly extend
    factor < 1.0 for special case of LJ benchmark so code will create
    correct-size stencil when there are 3 bins for every 5 lattice spacings
    */

    nextx = static_cast<int>(cutneigh * bininvx);

    if(nextx * binsizex < FACTOR * cutneigh) nextx++;

    nexty = static_cast<int>(cutneigh * bininvy);

    if(nexty * binsizey < FACTOR * cutneigh) nexty++;

    nextz = static_cast<int>(cutneigh * bininvz);

    if(nextz * binsizez < FACTOR * cutneigh) nextz++;
    if(team_neigh_build) {
      if((mbinx-2*nextx)%team_neigh_build) {
        check = 1;
        nbinx++;
      }
    }
  }

  nmax = (2 * nextz + 1) * (2 * nexty + 1) * (2 * nextx + 1);

  stencil = int_1d_view_type("Neighbor::stencil", nmax);
  int_1d_host_view_type h_stencil = Kokkos::create_mirror_view(stencil);
  nstencil = 0;
  int kstart = -nextz;

  if(halfneigh && ghost_newton) {
    kstart = 0;
    h_stencil[nstencil++] = 0;
  }

  for(k = kstart; k <= nextz; k++) {
    for(j = -nexty; j <= nexty; j++) {
      for(i = -nextx; i <= nextx; i++) {
        if(!ghost_newton || !halfneigh || (k > 0 || j > 0 || (j == 0 && i > 0)))
          if(bindist(i, j, k) < h_cutneighsq[0]) {
            h_stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
          }
      }
    }
  }

  Kokkos::deep_copy(stencil,h_stencil);
  mbins = mbinx * mbiny * mbinz;

  bincount = int_1d_view_type("Neighbor::bincount",mbins);
  bin_has_local = int_1d_view_type("Neighbor::bin_has_local",mbins);
  bin_list = int_1d_view_type("Neighbor::bin_list",mbins);
  bins = int_2d_view_type("Neighbor::bins",mbins , atoms_per_bin);

  return 0;
}

/* compute closest distance between central bin (0,0,0) and bin (i,j,k) */

MMD_float Neighbor::bindist(int i, int j, int k)
{
  MMD_float delx, dely, delz;

  if(i > 0)
    delx = (i - 1) * binsizex;
  else if(i == 0)
    delx = 0.0;
  else
    delx = (i + 1) * binsizex;

  if(j > 0)
    dely = (j - 1) * binsizey;
  else if(j == 0)
    dely = 0.0;
  else
    dely = (j + 1) * binsizey;

  if(k > 0)
    delz = (k - 1) * binsizez;
  else if(k == 0)
    delz = 0.0;
  else
    delz = (k + 1) * binsizez;

  return (delx * delx + dely * dely + delz * delz);
}
