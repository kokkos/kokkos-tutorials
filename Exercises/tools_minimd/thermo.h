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

#ifndef THERMO_H
#define THERMO_H

enum units {LJ, METAL};
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "timer.h"
#include "comm.h"
#include "types.h"

class Integrate;

class Thermo
{
  public:

    MMD_int nstat;
    MMD_int mstat;
    MMD_int ntimes;
    MMD_int* steparr;
    MMD_float* tmparr;
    MMD_float* engarr;
    MMD_float* prsarr;

    Thermo();
    ~Thermo();
    void setup(MMD_float, Integrate &integrate, Atom &atom, MMD_int);
    MMD_float temperature(Atom &);
    KOKKOS_INLINE_FUNCTION
    void operator() (const int& i, MMD_float& mv) const;

    MMD_float energy(Atom &, Neighbor &, Force*);
    MMD_float pressure(MMD_float, Force*);
    void compute(MMD_int, Atom &, Neighbor &, Force*, Timer &, Comm &);

    x_const_view_type v;
    MMD_float mass;

    MMD_float t_act, p_act, e_act;
    MMD_float t_scale, e_scale, p_scale, mvv2e, dof_boltz;

  private:
    MMD_float rho;
};

#endif
