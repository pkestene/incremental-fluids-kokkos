#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

//#include "shared/real_type.h"
//#include "shared/utils.h"

// make the compiler ignore an unused variable
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

// make the compiler ignore an unused function
#ifdef __GNUC__
#define UNUSED_FUNCTION __attribute__ ((unused))
#else
#define UNUSED_FUNCTION
#endif

using Device = Kokkos::DefaultExecutionSpace;

enum KokkosLayout {
  KOKKOS_LAYOUT_LEFT,
  KOKKOS_LAYOUT_RIGHT
};

// 2d scalar array
typedef Kokkos::View<double**, Device>   Array2d;
typedef Array2d::HostMirror              Array2dHost;

// 3d scalar array
typedef Kokkos::View<double***, Device>  Array3d;
typedef Array3d::HostMirror              Array3dHost;

// 2d array of uint8_t
typedef Kokkos::View<uint8_t**, Device>  Array2d_uchar;
typedef Array2d_uchar::HostMirror        Array2dHost_uchar;

// 3d array of uint8_t
typedef Kokkos::View<uint8_t***, Device>  Array3d_uchar;
typedef Array3d_uchar::HostMirror        Array3dHost_uchar;

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a prefered layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 *
 * These function will eventually disappear.
 * We still need then as long as parallel_reduce does not accept MDRange policy.
 */

/* 2D */

KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny)
{
  UNUSED(Nx);
  UNUSED(Ny);
  
#ifdef KOKKOS_ENABLE_CUDA
  j = index / Nx;
  i = index - j*Nx;
#else
  i = index / Ny;
  j = index - i*Ny;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i, int j, int Nx, int Ny)
{
  UNUSED(Nx);
  UNUSED(Ny);
#ifdef KOKKOS_ENABLE_CUDA
  return i + Nx*j; // left layout
#else
  return j + Ny*i; // right layout
#endif
}

/* 3D */

KOKKOS_INLINE_FUNCTION
void index2coord(int index,
                 int &i, int &j, int &k,
                 int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  int NxNy = Nx*Ny;
  k = index / NxNy;
  j = (index - k*NxNy) / Nx;
  i = index - j*Nx - k*NxNy;
#else
  int NyNz = Ny*Nz;
  i = index / NyNz;
  j = (index - i*NyNz) / Nz;
  k = index - j*Nz - i*NyNz;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i,  int j,  int k,
                int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  return i + Nx*j + Nx*Ny*k; // left layout
#else
  return k + Nz*j + Nz*Ny*i; // right layout
#endif
}

/**
 * A simple lambda functor to rest a Kokkos view (Array2d)
 */
void reset_view(Array2d& data)
{
  
  typedef Array2d::size_type size_type;

  Kokkos::parallel_for(data.dimension_0()*data.dimension_1(),
		       KOKKOS_LAMBDA (const size_type index)
		       {
			 int ix, iy;
			 index2coord(index,ix,iy,data.dimension_0(),data.dimension_1());

			 data(ix,iy)=0.0;
		       });
} // reset_view


#endif // KOKKOS_SHARED_H_
