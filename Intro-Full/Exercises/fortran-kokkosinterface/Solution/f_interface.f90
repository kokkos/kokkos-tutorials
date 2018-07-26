module f_interface_mod

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env
  use :: abi_mod

  implicit none

  public

  interface
    subroutine f_kokkos_initialize() &
      bind(c, name='c_kokkos_initialize')
      use, intrinsic :: iso_c_binding
      implicit none
    end subroutine f_kokkos_initialize
  end interface

  interface
    subroutine f_kokkos_finalize() &
      bind(c, name='c_kokkos_finalize')
      use, intrinsic :: iso_c_binding
      implicit none
    end subroutine f_kokkos_finalize
  end interface

  interface
    subroutine f_axpy_kokkos( alpha, nd_array_x, nd_array_y ) &
      bind(c, name="c_axpy_kokkos")
      use, intrinsic :: iso_c_binding
      use :: abi_mod
      implicit none
      real(c_double), intent(inout) :: alpha
      type(nd_array_t), intent(inout) :: nd_array_x
      type(nd_array_t), intent(inout) :: nd_array_y
    end subroutine f_axpy_kokkos
  end interface

  contains

  subroutine kokkos_initialize()
    use, intrinsic :: iso_c_binding
    implicit none
    call f_kokkos_initialize()
  end subroutine kokkos_initialize

  subroutine kokkos_finalize()
    use, intrinsic :: iso_c_binding
    implicit none
    call f_kokkos_finalize()
  end subroutine kokkos_finalize

  subroutine axpy_kokkos( alpha, array_x, array_y )

    use, intrinsic :: iso_c_binding
    use :: abi_mod

    implicit none

    real(c_double) :: alpha
    real(c_double), dimension(:), intent(inout) :: array_x
    real(c_double), dimension(:), intent(inout) :: array_y
    type(nd_array_t) :: nd_array_x
    type(nd_array_t) :: nd_array_y
    integer(c_size_t), target :: array_x_dims(1)
    integer(c_size_t), target :: array_y_dims(1)
    integer(c_size_t), target :: array_x_stride(1)
    integer(c_size_t), target :: array_y_stride(1)


    nd_array_x = to_nd_array( array_x, array_x_dims, array_x_stride )
    nd_array_y = to_nd_array( array_y, array_y_dims, array_y_stride )
    
    call f_axpy_kokkos( alpha, nd_array_x, nd_array_y )

  end subroutine axpy_kokkos

end module f_interface_mod
