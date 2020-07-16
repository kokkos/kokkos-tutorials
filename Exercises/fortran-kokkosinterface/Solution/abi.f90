module abi_mod

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  implicit none

  private

  public nd_array_t
  public to_nd_array

  type, bind(C) :: nd_array_t
    integer(c_size_t) :: rank
    type(c_ptr) :: dims
    type(c_ptr) :: strides
    type(c_ptr) :: data
  end type nd_array_t

  interface to_nd_array
    module procedure to_nd_array_r64_1d
  end interface

  contains

    function to_nd_array_r64_1d(array, dims, strides) result(ndarray)

      real(REAL64), target, intent(in) :: array(:)
      integer(c_size_t), target, intent(inout) :: dims(1)
      integer(c_size_t), target, intent(inout) :: strides(1)
      type(nd_array_t) :: ndarray

      dims(1) = size(array, 1, c_size_t)

      if (size(array, 1) .ge. 2) then
      strides(1) = &
      (transfer(c_loc(array(2)), 1_c_size_t) - &
      transfer(c_loc(array(1)), 1_c_size_t)) / c_sizeof(array(1))
      else
      strides(1) = 0
      end if

      ndarray%rank = 1
      ndarray%dims = c_loc(dims(1))
      ndarray%strides = c_loc(strides(1))
      ndarray%data = c_loc(array(1))

    end function to_nd_array_r64_1d

end module
