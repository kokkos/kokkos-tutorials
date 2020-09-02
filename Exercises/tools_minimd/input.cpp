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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "mpi.h"

#include "ljs.h"
#include "atom.h"
#include "force.h"
#include "neighbor.h"
#include "integrate.h"
#include "thermo.h"
#include "types.h"


#define MAXLINE 256

int input(In &in, const char* filename)
{
  FILE* fp;
  int flag;
  char line[MAXLINE];

  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  fp = fopen(filename, "r");

  if(fp == NULL) flag = 0;
  else flag = 1;

  if(flag == 0) {
    if(me == 0) printf("ERROR: Cannot open %s\n", filename);

    return 1;
  }

#if PRECISION==1
    fgets(line, MAXLINE, fp);
    fgets(line, MAXLINE, fp);
    fgets(line, MAXLINE, fp);

    if(strcmp(strtok(line, " \t\n"), "lj") == 0) in.units = 0;
    else if(strcmp(strtok(line, " \t\n"), "metal") == 0) in.units = 1;
    else {
      printf("Unknown units option in file at line 3 ('%s'). Expecting either 'lj' or 'metal'.\n", line);
      MPI_Finalize();
      exit(0);
    }

    fgets(line, MAXLINE, fp);

    if(strcmp(strtok(line, " \t\n"), "none") == 0) in.datafile = NULL;
    else {
      in.datafile = new char[1000];
      char* ptr = strtok(line, " \t");

      if(ptr == NULL) ptr = line;

      strcpy(in.datafile, ptr);
    }

    fgets(line, MAXLINE, fp);

    if(strcmp(strtok(line, " \t\n"), "lj") == 0) in.forcetype = FORCELJ;
    else if(strcmp(strtok(line, " \t\n"), "eam") == 0) in.forcetype = FORCEEAM;
    else {
      printf("Unknown forcetype option in file at line 5 ('%s'). Expecting either 'lj' or 'eam'.\n", line);
      MPI_Finalize();
      exit(0);
    }

    fgets(line, MAXLINE, fp);
    sscanf(line, "%e %e", &in.epsilon, &in.sigma);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%d %d %d", &in.nx, &in.ny, &in.nz);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%d", &in.ntimes);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%e", &in.dt);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%e", &in.t_request);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%e", &in.rho);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%d", &in.neigh_every);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%e %e", &in.force_cut, &in.neigh_cut);
    fgets(line, MAXLINE, fp);
    sscanf(line, "%d", &in.thermo_nstat);
    fclose(fp);
#else
#if PRECISION==2
      fgets(line, MAXLINE, fp);
      fgets(line, MAXLINE, fp);
      fgets(line, MAXLINE, fp);

      if(strcmp(strtok(line, " \t\n"), "lj") == 0) in.units = 0;
      else if(strcmp(line, "metal") == 0) in.units = 1;
      else {
        printf("Unknown units option in file at line 3 ('%s'). Expecting either 'lj' or 'metal'.\n", line);
        MPI_Finalize();
        exit(0);
      }

      fgets(line, MAXLINE, fp);

      if(strcmp(strtok(line, " \t\n"), "none") == 0) in.datafile = NULL;
      else {
        in.datafile = new char[1000];
        char* ptr = strtok(line, " \t");

        if(ptr == NULL) ptr = line;

        strcpy(in.datafile, ptr);
      }

      fgets(line, MAXLINE, fp);

      if(strcmp(strtok(line, " \t\n"), "lj") == 0) in.forcetype = FORCELJ;
      else if(strcmp(line, "eam") == 0) in.forcetype = FORCEEAM;
      else {
        printf("Unknown forcetype option in file at line 5 ('%s'). Expecting either 'lj' or 'eam'.\n", line);
        MPI_Finalize();
        exit(0);
      }

      fgets(line, MAXLINE, fp);
      sscanf(line, "%le %le", &in.epsilon, &in.sigma);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%d %d %d", &in.nx, &in.ny, &in.nz);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%d", &in.ntimes);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%le", &in.dt);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%le", &in.t_request);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%le", &in.rho);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%d", &in.neigh_every);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%le %le", &in.force_cut, &in.neigh_cut);
      fgets(line, MAXLINE, fp);
      sscanf(line, "%d", &in.thermo_nstat);
      fclose(fp);
#else
      if(me == 0)
        printf("Invalid MMD_float size specified: crash imminent.\n");
#endif
#endif

  in.neigh_cut += in.force_cut;
  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}
