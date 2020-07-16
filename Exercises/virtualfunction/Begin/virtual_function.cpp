#include<classes.hpp>

// Exercise
// 1. Launch a parallel kernel an use placement new to create virtual objects on
//    device
// 2. Launch a parallel kernel to destroy the virtual objects on the device before
//    freeing the memory

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);

  {
    Foo* f_1 = (Foo*) Kokkos::kokkos_malloc(sizeof(Foo_1));
    Foo* f_2 = (Foo*) Kokkos::kokkos_malloc(sizeof(Foo_2));

    Kokkos::parallel_for("CreateObjects",1, KOKKOS_LAMBDA (const int&) {
      // TODO placement new Foo_1 in f_1 and Foo_2 in f_2
    });

    int value_1,value_2;
    Kokkos::parallel_reduce("CheckValues",1, KOKKOS_LAMBDA (const int&, int& lsum) {
      lsum = f_1->value();
    },value_1);

    Kokkos::parallel_reduce("CheckValues",1, KOKKOS_LAMBDA (const int&, int& lsum) {
      lsum = f_2->value();
    },value_2);

    printf("Values: %i %i\n",value_1,value_2);

    Kokkos::parallel_for("DestroyObjects",1, KOKKOS_LAMBDA (const int&) {
      // TODO destroy f_1 and f_2
    });

    Kokkos::kokkos_free(f_1);
    Kokkos::kokkos_free(f_2);
  }

  Kokkos::finalize();
}
