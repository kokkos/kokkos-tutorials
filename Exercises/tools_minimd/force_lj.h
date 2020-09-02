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

#ifndef FORCELJ_H
#define FORCELJ_H

#include "atom.h"
#include "neighbor.h"
#include "types.h"
#include "force.h"
#include "comm.h"


class ForceLJ : Force
{
  public:

    typedef eng_virial_type value_type;

    template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
    struct TagComputeHalfNeighThread {
      enum {evflag = EVFLAG};
      enum {ghost_newton = GHOST_NEWTON};
      enum {stack_params = STACK_PARAMS};
    };
    template<int EVFLAG, int STACK_PARAMS>
    struct TagComputeFullNeigh {
      enum {evflag = EVFLAG};
      enum {stack_params = STACK_PARAMS};
    };

    ForceLJ(int ntypes_);
    virtual ~ForceLJ();
    void setup();
    void compute(Atom &, Neighbor &, Comm &, int);

  protected:

    template<int EVFLAG>
    void compute_original(Atom &, Neighbor &, int);
    template<int EVFLAG, int GHOST_NEWTON>
    void compute_halfneigh(Atom &, Neighbor &, int);

    template<int EVFLAG, int GHOST_NEWTON>
    void compute_halfneigh_threaded(Atom &, Neighbor &, int);

    template<int EVFLAG>
    void compute_fullneigh(Atom &, Neighbor &, int);

  public:
    template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> , const int& i ) const;
    template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> , const int& i, eng_virial_type& eng_virial ) const;

    template<int EVFLAG, int STACK_PARAMS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeFullNeigh<EVFLAG,STACK_PARAMS> , const int& i ) const;
    template<int EVFLAG, int STACK_PARAMS>
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeFullNeigh<EVFLAG,STACK_PARAMS> , const int& i, eng_virial_type& eng_virial ) const;

};

#endif
