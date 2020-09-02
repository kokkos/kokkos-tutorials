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


#ifndef FORCEEAM_H
#define FORCEEAM_H

#include "stdio.h"
#include "atom.h"
#include "neighbor.h"
#include "types.h"
#include "mpi.h"
#include "comm.h"
#include "force.h"

class ForceEAM : Force
{
  public:

    typedef eng_virial_type value_type;


    struct TagHalfNeighInitial {};
    template<int EVFLAG>
    struct TagHalfNeighMiddle {};
    template<int EVFLAG>
    struct TagHalfNeighFinal {};

    template<int EVFLAG>
    struct TagFullNeighInitial {};
    template<int EVFLAG>
    struct TagFullNeighFinal {};

    struct TagEAMPackComm {};
    struct TagEAMUnpackComm {};

    typedef Kokkos::DualView<MMD_float**[7],Kokkos::LayoutRight> spline_dv_type;
    typedef spline_dv_type::t_host spline_host_type;
    typedef Kokkos::View<const MMD_float**[7],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::RandomAccess> > const_spline_type;

    // public variables so USER-ATC package can access them

    MMD_float cutmax;

    // potentials as array data

    MMD_int nrho, nr;
    MMD_int nrho_tot, nr_tot;
    MMD_float* frho, *rhor, *z2r;

    // potentials in spline form used for force computation

    MMD_float dr, rdr, drho, rdrho;
    const_spline_type rhor_spline, frho_spline, z2r_spline;

    int first, iswap;
    float_1d_view_type buf;
    int_2d_lr_view_type sendlist;

    ForceEAM(int ntypes_);
    virtual ~ForceEAM();
    virtual void compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    virtual void coeff(const char*);
    virtual void setup();
    void init_style();

    virtual MMD_int pack_comm(int n, int iswap_, float_1d_view_type abuf, const int_2d_lr_view_type& asendlist);
    virtual void unpack_comm(int n, int first_, float_1d_view_type abuf);
    MMD_int pack_reverse_comm(MMD_int, MMD_int, float_1d_view_type);
    void unpack_reverse_comm(MMD_int, MMD_int*, float_1d_view_type);
    MMD_float memory_usage();

    KOKKOS_INLINE_FUNCTION
    void operator() (TagHalfNeighInitial , const int& i ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagHalfNeighMiddle<0> , const int& i ) const;
    template<int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagHalfNeighMiddle<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagHalfNeighFinal<0> , const int& i ) const;
    template<int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagHalfNeighFinal<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const;

    KOKKOS_INLINE_FUNCTION
    void operator() (TagFullNeighInitial<0> , const int& i ) const;
    template<int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagFullNeighInitial<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagFullNeighFinal<0> , const int& i) const;
    template<int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagFullNeighFinal<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const;

    KOKKOS_INLINE_FUNCTION
    void operator() (TagEAMPackComm, const int& i) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagEAMUnpackComm, const int& i) const;

  protected:
    void compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    void compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me);

    // per-atom arrays

    float_1d_view_type rho, fp;
    float_1d_atomic_view_type rho_a;

    MMD_int nmax;

    // potentials as file data

    MMD_int* map;                   // which element each atom type maps to

    struct Funcfl {
      char* file;
      MMD_int nrho, nr;
      double drho, dr, cut, mass;
      MMD_float* frho, *rhor, *zr;
    };
    Funcfl funcfl;

    void array2spline();
    void interpolate(MMD_int n, MMD_float delta, MMD_float* f, spline_host_type spline);
    void grab(FILE*, MMD_int, MMD_float*);

    virtual void read_file(const char*);
    virtual void file2array();

    void bounds(char* str, int nmax, int &nlo, int &nhi);

    void communicate(Atom &atom, Comm &comm);
};



#endif
