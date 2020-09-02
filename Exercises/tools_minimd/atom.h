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

#ifndef ATOM_H
#define ATOM_H

#include "types.h"
#include "Kokkos_Sort.hpp"

class Neighbor;
struct Box {
  MMD_float xprd, yprd, zprd;
  MMD_float xlo, xhi;
  MMD_float ylo, yhi;
  MMD_float zlo, zhi;
};

class Atom
{
  public:

    struct TagAtomPBC {};
    struct TagAtomPackCommPBC {};
    struct TagAtomPackCommNoPBC {};
    struct TagAtomPackCommSelfPBC {};
    struct TagAtomPackCommSelfNoPBC {};
    struct TagAtomUnpackComm {};
    struct TagAtomPackReverse {};
    struct TagAtomUnpackReverse {};
    struct TagAtomSort {};

    typedef int value_type;
    int natoms;
    int nlocal, nghost;
    int nmax;

    x_view_type x;
    x_view_type v;
    x_view_type f;
    x_host_view_type h_x;
    x_host_view_type h_v;
    x_host_view_type h_f;


    int ntypes;
    int_1d_view_type type, new_type, old_type;
    int_1d_host_view_type h_type;

    x_view_type xold, new_x, new_v, old_x, old_v;

    MMD_float virial, mass;

    int comm_size, reverse_size, border_size;

    Box box;

    Atom() {};
    Atom(int ntypes_);
    ~Atom();

    void operator= (const Atom& src) {
      natoms = src.natoms;
      nlocal = src.nlocal;
      nghost = src.nghost;
      nmax = src.nmax;
      type = src.type;
      x = src.x;
      v = src.v;
      f = src.f;
      ntypes = src.ntypes;
      xold = src.xold;
      new_type = src.new_type;
      new_x = src.new_x;
      new_v = src.new_v;
      old_type = src.old_type;
      old_x = src.old_x;
      old_v = src.old_v;
      virial = src.virial;
      mass = src.mass;
      comm_size = src.comm_size;
      reverse_size = src.reverse_size;
      border_size = src.border_size;

      binpos = src.binpos;
      bins = src.bins;
      x_copy = src.x_copy;
      v_copy = src.v_copy;
      type_copy = src.type_copy;
      copy_size = src.copy_size;

      buf = src.buf;
      list = src.list;
      pbc_flags[0] = src.pbc_flags[0];
      pbc_flags[1] = src.pbc_flags[1];
      pbc_flags[2] = src.pbc_flags[2];
      pbc_flags[3] = src.pbc_flags[3];
      first = src.first;

      box = src.box;
    }

    void addatom(MMD_float, MMD_float, MMD_float, MMD_float, MMD_float, MMD_float);

    void pbc();
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomPBC, const int& i) const;

    void growarray();

    KOKKOS_INLINE_FUNCTION
    void copy(int, int) const;

    void pack_comm(int, int_1d_view_type , float_1d_view_type, int*);
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomPackCommPBC, const int& i) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomPackCommNoPBC, const int& i) const;

    void unpack_comm(int, int, float_1d_view_type);
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomUnpackComm, const int& i) const;

    void pack_comm_self(int n, int_1d_view_type list_in, int first_in, int* pbc_flags_in);
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomPackCommSelfPBC, const int& i) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomPackCommSelfNoPBC, const int& i) const;

    void pack_reverse(int, int, float_1d_view_type);
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomPackReverse, const int& i) const;

    void unpack_reverse(int, int_1d_view_type , float_1d_view_type);

    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomUnpackReverse, const int& i) const;

    KOKKOS_INLINE_FUNCTION
    int pack_border(int, MMD_float*, const int*) const;
    KOKKOS_INLINE_FUNCTION
    int unpack_border(int, MMD_float*) const;
    KOKKOS_INLINE_FUNCTION
    int pack_exchange(int, MMD_float*) const;
    KOKKOS_INLINE_FUNCTION
    int unpack_exchange(int, MMD_float*) const;
    KOKKOS_INLINE_FUNCTION
    int skip_exchange(MMD_float*);

    void destroy_2d_MMD_float_array(MMD_float*);

    void destroy_1d_int_array(int*);

    void sort(Neighbor & neighbor);
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomSort, const int& i) const;

    KOKKOS_INLINE_FUNCTION
    void operator() (TagAtomSort, const int& i, int& sum, bool final) const;

  private:
    int_1d_view_type binpos;
    int_2d_view_type bins;
    x_view_type x_copy;
    x_view_type v_copy;
    int_1d_view_type type_copy;
    int copy_size;

    float_1d_view_type buf;
    int_1d_view_type list;
    int pbc_flags[4];
    int first;
};

struct MiniMDBinOp3D {
  int max_bins_[3];
  double mul_[3];
  MMD_float range_[3];
  MMD_float min_[3];

  MiniMDBinOp3D(int max_bins[], MMD_float min[], MMD_float max[] )
  {
    max_bins_[0] = max_bins[0]+1;
    max_bins_[1] = max_bins[1]+1;
    max_bins_[2] = max_bins[2]+1;
    mul_[0] = 1.0*max_bins[0]/(max[0]-min[0]);
    mul_[1] = 1.0*max_bins[1]/(max[1]-min[1]);
    mul_[2] = 1.0*max_bins[2]/(max[2]-min[2]);
    range_[0] = max[0]-min[0];
    range_[1] = max[1]-min[1];
    range_[2] = max[2]-min[2];
    min_[0] = min[0];
    min_[1] = min[1];
    min_[2] = min[2];
  }

  template<class ViewType>
  KOKKOS_INLINE_FUNCTION
  int bin(ViewType& keys, const int& i) const {
    return int( (((int(mul_[2]*(keys(i,2)-min_[2]))*max_bins_[1]) +
                   int(mul_[1]*(keys(i,1)-min_[1])))*max_bins_[0]) +
                   int(mul_[0]*(keys(i,0)-min_[0])));
  }

  KOKKOS_INLINE_FUNCTION
  int max_bins() const {
    return max_bins_[0]*max_bins_[1]*max_bins_[2];
  }

  template<class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION
  bool operator()(ViewType& keys, iType1& i1 , iType2& i2) const {
    if (keys(i1,0)>keys(i2,0)) return true;
    else if (keys(i1,0)==keys(i2,0)) {
      if (keys(i1,1)>keys(i2,1)) return true;
      else if (keys(i1,1)==keys(i2,2)) {
        if (keys(i1,2)>keys(i2,2)) return true;
      }
    }
    return false;
  }
};
KOKKOS_INLINE_FUNCTION
void Atom::copy(int i, int j) const
{
  x(j,0) = x(i,0);
  x(j,1) = x(i,1);
  x(j,2) = x(i,2);
  v(j,0) = v(i,0);
  v(j,1) = v(i,1);
  v(j,2) = v(i,2);
  type[j] = type[i];
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomPBC, const int& i) const {
  if(x(i,0) < 0.0) x(i,0) += box.xprd;

  if(x(i,0) >= box.xprd) x(i,0) -= box.xprd;

  if(x(i,1) < 0.0) x(i,1) += box.yprd;

  if(x(i,1) >= box.yprd) x(i,1) -= box.yprd;

  if(x(i,2) < 0.0) x(i,2) += box.zprd;

  if(x(i,2) >= box.zprd) x(i,2) -= box.zprd;
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomPackCommPBC, const int& i) const {
  const int j = list[i];
  buf[3 * i] = x(j,0) + pbc_flags[1] * box.xprd;
  buf[3 * i + 1] = x(j,1) + pbc_flags[2] * box.yprd;
  buf[3 * i + 2] = x(j,2) + pbc_flags[3] * box.zprd;
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomPackCommNoPBC, const int& i) const {
  const int j = list[i];
  buf[3 * i] = x(j,0);
  buf[3 * i + 1] = x(j,1);
  buf[3 * i + 2] = x(j,2);
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomUnpackComm, const int& i) const {
  x((first + i) , 0) = buf[3 * i];
  x((first + i) , 1) = buf[3 * i + 1];
  x((first + i) , 2) = buf[3 * i + 2];
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomPackCommSelfPBC, const int& i) const {
  const int j = list[i];
  x((first + i) , 0) = x(j,0) + pbc_flags[1] * box.xprd;
  x((first + i) , 1) = x(j,1) + pbc_flags[2] * box.yprd;
  x((first + i) , 2) = x(j,2) + pbc_flags[3] * box.zprd;
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomPackCommSelfNoPBC, const int& i) const {
  const int j = list[i];
  x((first + i) , 0) = x(j,0);
  x((first + i) , 1) = x(j,1);
  x((first + i) , 2) = x(j,2);
}


KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomPackReverse, const int& i) const {
  buf[3 * i] = f((first + i) , 0);
  buf[3 * i + 1] = f((first + i) , 1);
  buf[3 * i + 2] = f((first + i) , 2);
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomUnpackReverse, const int& i) const {
  const int j = list[i];
  f(j,0) += buf[3 * i];
  f(j,1) += buf[3 * i + 1];
  f(j,2) += buf[3 * i + 2];
}

KOKKOS_INLINE_FUNCTION
int Atom::pack_border(int i, MMD_float* buf, const int* pbc_flags) const
{
  int m = 0;

  if(pbc_flags[0] == 0) {
    buf[m++] = x(i,0);
    buf[m++] = x(i,1);
    buf[m++] = x(i,2);
    buf[m++] = type[i];
  } else {
    buf[m++] = x(i,0) + box.xprd * pbc_flags[1];
    buf[m++] = x(i,1) + box.yprd * pbc_flags[2];
    buf[m++] = x(i,2) + box.zprd * pbc_flags[3];
    buf[m++] = type[i];
  }

  return m;
}

KOKKOS_INLINE_FUNCTION
int Atom::unpack_border(int i, MMD_float* buf) const
{
  int m = 0;
  x(i,0) = buf[m++];
  x(i,1) = buf[m++];
  x(i,2) = buf[m++];
  type[i] = buf[m++];
  return m;
}

KOKKOS_INLINE_FUNCTION
int Atom::pack_exchange(int i, MMD_float* buf) const
{
  int m = 0;
  buf[m++] = x(i,0);
  buf[m++] = x(i,1);
  buf[m++] = x(i,2);
  buf[m++] = v(i,0);
  buf[m++] = v(i,1);
  buf[m++] = v(i,2);
  buf[m++] = type[i];
  return m;
}

KOKKOS_INLINE_FUNCTION
int Atom::unpack_exchange(int i, MMD_float* buf) const
{
  int m = 0;
  x(i,0) = buf[m++];
  x(i,1) = buf[m++];
  x(i,2) = buf[m++];
  v(i,0) = buf[m++];
  v(i,1) = buf[m++];
  v(i,2) = buf[m++];
  type[i] = buf[m++];
  return m;
}

KOKKOS_INLINE_FUNCTION
int Atom::skip_exchange(MMD_float* buf)
{
  return 7;
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomSort, const int& i, int& sum, bool final) const {
  sum += binpos[i];
  if(final)
    binpos[i] = sum;
}

KOKKOS_INLINE_FUNCTION
void Atom::operator() (TagAtomSort, const int& mybin) const {
  const int start = mybin>0?binpos[mybin-1]:0;
  const int count = binpos[mybin] - start;
  for(int k=0; k<count; k++) {
    const int new_i = start+k;
    const int old_i = bins(mybin, k);
    new_x(new_i,0) = old_x(old_i,0);
    new_x(new_i,1) = old_x(old_i,1);
    new_x(new_i,2) = old_x(old_i,2);
    new_v(new_i,0) = old_v(old_i,0);
    new_v(new_i,1) = old_v(old_i,1);
    new_v(new_i,2) = old_v(old_i,2);
    new_type[new_i] = old_type[old_i];
  }
}
#endif
