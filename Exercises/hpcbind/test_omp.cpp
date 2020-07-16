#include <cstdio>
#include <cstdlib>

#include <omp.h>
#include <hwloc.h>

namespace {

hwloc_topology_t g_topology;

void cpuset_snprintf( char *out, size_t n, hwloc_const_cpuset_t cpuset)
{
  char buffer[64];

  hwloc_bitmap_list_snprintf( buffer, sizeof(buffer), cpuset );

  snprintf( out, n
          , "pu: %s"
          , buffer
          );
}

void where_am_i()
{
  hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
  hwloc_get_last_cpu_location( g_topology, cpuset, HWLOC_CPUBIND_THREAD );

  int os_pu_index = hwloc_bitmap_first( cpuset );

  hwloc_bitmap_free(cpuset);

  hwloc_obj_t obj  = hwloc_get_pu_obj_by_os_index( g_topology
                                                 , os_pu_index
                                                 );
  if (obj) {
    const int pu_logical_index = obj->logical_index;

    while ( obj && obj->type != HWLOC_OBJ_CORE ) {
      obj = obj->parent;
    }

    const int core_logical_index = obj ?  (int)obj->logical_index : -1;

    const int thread_num  = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();
    const int level       = omp_get_level();

    if (level == 1) {
    printf( "%3d of %3d : core: %3d pu: %3d\n"
          , thread_num
          , num_threads
          , core_logical_index
          , pu_logical_index
          );
    } else if ( level == 2 ) {

      const int parent = omp_get_ancestor_thread_num(1);
      const int num_parents = omp_get_team_size(1);

      printf( "%3d of %3d : %3d of %3d : core: %3d pu: %3d\n"
          , parent
          , num_parents
          , thread_num
          , num_threads
          , core_logical_index
          , pu_logical_index
          );
    }
  }
  else {
    printf("Unable to detect binding\n");
  }
}

} // namespace

int main( int argc, char * argv[] )
{
  hwloc_topology_init( &g_topology );
  hwloc_topology_load( g_topology );

  const int nested = omp_get_nested();

  if (!nested) {
    #pragma omp parallel
    {
      where_am_i();
    }

  } else {

    int nthreads = omp_get_max_threads();
    #pragma omp parallel num_threads(nthreads/2)
    {
      #pragma omp parallel num_threads(2)
      {
        where_am_i();
      }
    }
  }

  hwloc_topology_destroy( g_topology );

  return 0;
}

