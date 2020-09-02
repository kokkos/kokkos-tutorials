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
#include "string.h"
#include "stdlib.h"
#include "mpi.h"
#include "atom.h"
#include "neighbor.h"

#define DELTA 20000

Atom::Atom(int ntypes_)
{
  natoms = 0;
  nlocal = 0;
  nghost = 0;
  nmax = 0;
  copy_size = 0;

  comm_size = 3;
  reverse_size = 3;
  border_size = 4;

  mass = 1;

  ntypes = ntypes_;
}

Atom::~Atom()
{
}

void Atom::growarray()
{
  nmax += DELTA;
  Kokkos::resize(x,nmax);
  Kokkos::resize(v,nmax);
  Kokkos::resize(f,nmax);
  Kokkos::resize(type,nmax);
  Kokkos::resize(xold,nmax);
  h_x = Kokkos::create_mirror_view(x);
  h_v = Kokkos::create_mirror_view(v);
  h_type = Kokkos::create_mirror_view(type);
}

void Atom::addatom(MMD_float x_in, MMD_float y_in, MMD_float z_in,
                   MMD_float vx_in, MMD_float vy_in, MMD_float vz_in)
{
  if(nlocal == nmax) {
    Kokkos::deep_copy(x,h_x);
    Kokkos::deep_copy(v,h_v);
    Kokkos::deep_copy(type,h_type);
    growarray();
    Kokkos::deep_copy(h_x,x);
    Kokkos::deep_copy(h_v,v);
    Kokkos::deep_copy(h_type,type);
  }

  h_x(nlocal,0) = x_in;
  h_x(nlocal,1) = y_in;
  h_x(nlocal,2) = z_in;
  h_v(nlocal,0) = vx_in;
  h_v(nlocal,1) = vy_in;
  h_v(nlocal,2) = vz_in;
  h_type[nlocal] = rand()%ntypes;

  nlocal++;
}

/* enforce PBC
   order of 2 tests is important to insure lo-bound <= coord < hi-bound
   even with round-off errors where (coord +/- epsilon) +/- period = bound */

void Atom::pbc()
{
  Kokkos::parallel_for("Atom::pbc",Kokkos::RangePolicy<TagAtomPBC>(0,nlocal), *this);
}

void Atom::pack_comm(int n, int_1d_view_type list_in, float_1d_view_type buf_in, int* pbc_flags_in)
{
  list = list_in;
  buf = buf_in;
  for(int i = 0; i < 4; i++) pbc_flags[i] = pbc_flags_in[i];

  if(pbc_flags[0] == 0) {
    Kokkos::parallel_for("Comm::pack",Kokkos::RangePolicy<TagAtomPackCommNoPBC>(0,n), *this);
  } else {
    Kokkos::parallel_for("Comm::pack",Kokkos::RangePolicy<TagAtomPackCommPBC>(0,n), *this);
  }
}

void Atom::unpack_comm(int n, int first_in, float_1d_view_type buf_in)
{
  first = first_in;
  buf = buf_in;
  Kokkos::parallel_for("Comm::unpack",Kokkos::RangePolicy<TagAtomUnpackComm>(0,n), *this);
}

void Atom::pack_comm_self(int n, int_1d_view_type list_in, int first_in, int* pbc_flags_in)
{
  list = list_in;
  first = first_in;
  for(int i = 0; i < 4; i++) pbc_flags[i] = pbc_flags_in[i];

  if(pbc_flags[0] == 0) {
    Kokkos::parallel_for("Comm::self",Kokkos::RangePolicy<TagAtomPackCommSelfNoPBC>(0,n), *this);
  } else {
    Kokkos::parallel_for("Comm::self",Kokkos::RangePolicy<TagAtomPackCommSelfPBC>(0,n), *this);
  }
}

void Atom::pack_reverse(int n, int first_in, float_1d_view_type buf_in)
{
  first = first_in;
  buf = buf_in;
  Kokkos::parallel_for(Kokkos::RangePolicy<TagAtomPackReverse>(0,n), *this);
}

void Atom::unpack_reverse(int n, int_1d_view_type list_in, float_1d_view_type buf_in)
{
  list = list_in;
  buf = buf_in;
  Kokkos::parallel_for(Kokkos::RangePolicy<TagAtomUnpackReverse>(0,n), *this);
}



/* realloc a 2-d MMD_float array */

void Atom::sort(Neighbor &neighbor)
{

  //The following Kokkos sort works and is not slower than the handrolled one, but it does
  //produce a different order than what is used in the neighboring process resulting in unoptimal
  //locality for that.
  /*int bin_max[3] = {neighbor.mbinx,neighbor.mbiny,neighbor.mbinz};
  MMD_float min[3] = {box.xlo-neighbor.cutneigh,box.ylo-neighbor.cutneigh,box.zlo-neighbor.cutneigh};
  MMD_float max[3] = {box.xhi+neighbor.cutneigh,box.yhi+neighbor.cutneigh,box.zhi+neighbor.cutneigh};

  x_view_type x_local = Kokkos::subview(x,std::pair<int,int>(0,nlocal),Kokkos::ALL());
  x_view_type v_local = Kokkos::subview(v,std::pair<int,int>(0,nlocal),Kokkos::ALL());
  int_1d_view_type type_local = Kokkos::subview(type,std::pair<int,int>(0,nlocal));
  typedef Kokkos::SortImpl::DefaultBinOp3D<x_view_type> BinOp;

  BinOp bin_op(bin_max,min,max);
  Kokkos::BinSort< x_view_type , BinOp >
    Sorter(x_local,bin_op,false);
  Sorter.create_permute_vector();
  Sorter.template sort< x_view_type >(x_local);
  Sorter.template sort< x_view_type >(v_local);
  Sorter.template sort< int_1d_view_type >(type_local);

  return;*/

  neighbor.binatoms(*this,nlocal);

  Kokkos::fence();

  binpos = neighbor.bincount;
  bins = neighbor.bins;

  const int mbins = neighbor.mbins;

  Kokkos::parallel_scan("Atom::sort_scan",Kokkos::RangePolicy<TagAtomSort>(0,mbins), *this);

  if(copy_size<nmax) {
    x_copy = x_view_type("atom::x_copy",nmax);
    v_copy = x_view_type("atom::v_copy",nmax);
    type_copy = int_1d_view_type("atom::type_copy",nmax);
    copy_size = nmax;
  }

  new_x = x_copy;
  new_v = v_copy;
  new_type = type_copy;
  old_x = x;
  old_v = v;
  old_type = type;

  Kokkos::parallel_for("Atom::sort_for",Kokkos::RangePolicy<TagAtomSort>(0,mbins), *this);
  Kokkos::fence();

  x_view_type x_tmp = x;
  x_view_type v_tmp = v;
  int_1d_view_type type_tmp = type;

  x = x_copy;
  v = v_copy;
  type = type_copy;
  x_copy = x_tmp;
  v_copy = v_tmp;
  type_copy = type_tmp;
}


