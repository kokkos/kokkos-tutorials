/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// EXERCISE (scroll down to main() on line 157)
//   - Goal: Run Distance-1 and Distance-2 graph coloring on a small rectangular grid.

#include <vector>
#include <cstdio>
#include <cmath>
#include <sstream>
#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosGraph_Distance1Color.hpp"
#include "KokkosGraph_Distance2Color.hpp"

//Typedefs

using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using DeviceSpace = typename ExecSpace::memory_space;
using Kokkos::HostSpace;
using RowmapType = Kokkos::View<Offset*, DeviceSpace>;
using ColindsType = Kokkos::View<Ordinal*, DeviceSpace>;
using Handle  = KokkosKernels::Experimental::
  KokkosKernelsHandle<Offset, Ordinal, default_scalar, ExecSpace, DeviceSpace, DeviceSpace>;

namespace ColoringExercise
{
  constexpr Ordinal gridX = 15;
  constexpr Ordinal gridY = 25;
  constexpr Ordinal numVertices = gridX * gridY;

  //Helper to get the vertex ID given grid coordinates
  Ordinal getVertexID(Ordinal x, Ordinal y)
  {
    return y * gridX + x;
  }

  //Inverse of getVertexID
  void getVertexPos(Ordinal vert, Ordinal& x, Ordinal& y)
  {
    x = vert % gridX;
    y = vert / gridX;
  }

  //Helper to print out colors in the shape of the grid
  template<typename ColorView>
  void printColoring(ColorView colors, Ordinal numColors)
  {
    //Read colors on host
    auto colorsHost = Kokkos::create_mirror_view_and_copy(HostSpace(), colors);
    int numDigits = ceil(log10(numColors + 1));
    //Print out the grid, with columns aligned and at least one space between numbers
    std::ostringstream numFmtStream;
    numFmtStream << '%' << numDigits + 1 << 'd';
    std::string numFmt = numFmtStream.str();
    for(Ordinal y = 0; y < gridY; y++)
    {
      for(Ordinal x = 0; x < gridX; x++)
      {
        Ordinal vertex = getVertexID(x, y);
        int color = colorsHost(vertex);
        printf(numFmt.c_str(), color);
      }
      putchar('\n');
    }
  }

  //Build the graph on host, allocate these views on device and copy the graph to them.
  //Both rowmapDevice and colindsDevice are output parameters and should default-initialized (empty) on input.
  void generate9pt(RowmapType& rowmapDevice, ColindsType& colindsDevice)
  {
    //Generate the graph on host (use std::vector to not need to know
    //how many entries ahead of time)
    std::vector<Offset> rowmap(numVertices + 1);
    std::vector<Ordinal> colinds;
    rowmap[0] = 0;
    for(Ordinal vert = 0; vert < numVertices; vert++)
    {
      Ordinal x, y;
      getVertexPos(vert, x, y);
      //Loop over the neighbors in a 3x3 region
      for(Ordinal ny = y - 1; ny <= y + 1; ny++)
      {
        for(Ordinal nx = x - 1; nx <= x + 1; nx++)
        {
          //exclude the edge to self
          if(nx == x && ny == y)
            continue;
          //exclude vertices that would be outside the grid
          if(nx < 0 || nx >= gridX || ny < 0 || ny >= gridY)
            continue;
          //add the neighbor to colinds, forming an edge
          colinds.push_back(getVertexID(nx, ny));
        }
      }
      //mark where the current row ends
      rowmap[vert + 1] = colinds.size();
    }
    Offset numEdges = colinds.size();
    //Now that the graph is formed, copy rowmap and colinds to Kokkos::Views in device memory
    //The nonowning host views just alias the std::vectors.
    Kokkos::View<Offset*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> rowmapHost(rowmap.data(), numVertices + 1);
    Kokkos::View<Ordinal*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> colindsHost(colinds.data(), numEdges);
    //Allocate owning views on device with the correct size.
    rowmapDevice = RowmapType("Rowmap", numVertices + 1);
    colindsDevice = ColindsType("Colinds", numEdges);
    //Copy the graph from host to device
    Kokkos::deep_copy(rowmapDevice, rowmapHost);
    Kokkos::deep_copy(colindsDevice, colindsHost);
  }
}

int main(int argc, char* argv[])
{
  Kokkos::initialize();
  {
    using ColoringExercise::numVertices;
    RowmapType rowmapDevice;
    ColindsType colindsDevice;
    //Step 1: Generate the graph on host, then allocate space and copy to device.
    ColoringExercise::generate9pt(rowmapDevice, colindsDevice);
    //Step 2: Create handle and run distance-1 coloring.
    {
      using D1_ColoringHandleType = typename Handle::GraphColoringHandleType;
      using color_view_t = typename D1_ColoringHandleType::color_view_t;
      //General handle
      Handle handle;
//- EXERCISE: Create distance-1 graph coloring subhandle, with the default algorithm

//- EXERCISE: Run coloring on the graph (rowmapDevice, colindsDevice), which has numVertices rows/columns

      D1_ColoringHandleType* colorHandle = handle.get_graph_coloring_handle();
//- EXERCISE: From colorHandle, get the color array and the number of colors.
      color_view_t colors /* = colorHandle-> ... */;
      Ordinal numColors /* = colorHandle-> ... */;
      printf("9-pt stencil: Distance-1 Colors (used %d):\n", (int) numColors);
      ColoringExercise::printColoring(colors, numColors);
      putchar('\n');
//- EXERCISE: Clean up the distance-1 graph coloring subhandle

    }
    //Step 3: Create handle and run distance-2 coloring.
    {
      using D2_ColoringHandleType = typename Handle::GraphColorDistance2HandleType;
      using color_view_t = typename D2_ColoringHandleType::color_view_type;
      //General handle
      Handle handle;
//- EXERCISE: Create distance-2 graph coloring subhandle, with the default algorithm

//- EXERCISE: Run D2 coloring on the graph (rowmapDevice, colindsDevice), which has numVertices vertices.

      D2_ColoringHandleType* colorHandleD2 = handle.get_distance2_graph_coloring_handle();
//- EXERCISE: From colorHandle, get the color array and the number of colors.
      color_view_t colors /* = colorHandleD2-> ... */;
      Ordinal numColors /* = colorHandleD2-> ... */;
      printf("9-pt stencil: Distance-2 Colors (used %d):\n", (int) numColors);
      ColoringExercise::printColoring(colors, numColors);
      putchar('\n');
//- EXERCISE: Clean up the distance-2 graph coloring subhandle

    }
  }
  Kokkos::finalize();
  return 0;
}

