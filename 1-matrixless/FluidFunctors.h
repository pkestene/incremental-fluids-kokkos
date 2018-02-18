#ifndef FLUID_FUNCTORS_H_
#define FLUID_FUNCTORS_H_

#include "kokkos_shared.h"

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "FluidQuantity.h"
#include "FluidSolver.h"

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Advection update functor.
 */
class AdvectionFunctor
{

public:
    
  /**
   * \param[out] data is a scalar quantity array to advect
   * \param[in] u velocity along x for advection
   * \param[in] v velocity along y for advection
   */
  AdvectionFunctor(FluidQuantity data,
		   FluidQuantity u,
		   FluidQuantity v,
		   double timestep) :
    data(data),
    u(u),
    v(v),
    timestep(timestep),
    _w(data._w),
    _h(data._h),
    _ox(data._ox),
    _oy(data._oy),
    _hx(data._hx)
  {
  };
    
  
  // Simple forward Euler method for velocity integration in time
  KOKKOS_INLINE_FUNCTION
  void euler(double &x,
	     double &y) const {
    
    double uVel = u.lerp(x, y)/_hx;
    double vVel = v.lerp(x, y)/_hx;
    
    x -= uVel*timestep;
    y -= vVel*timestep;
    
  } // euler
  
  // Advect grid in velocity field u, v with given timestep
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {
    
    int ix, iy;
    index2coord(index,ix,iy,_w,_h);
    
    //for (int iy = 0, idx = 0; iy < _h; iy++) {
    //for (int ix = 0;          ix < _w; ix++, idx++) {
    double x = ix + _ox;
    double y = iy + _oy;
    
    // First component: Integrate in time, update x and y
    euler(x, y);
    
    // Second component: Interpolate from grid
    data._dst(ix,iy) = data.lerp(x, y);
    
  } // advection functor - operator()
  
  FluidQuantity data;
  FluidQuantity u;
  FluidQuantity v;
  double timestep;
  int _w;
  int _h;
  double _ox;
  double _oy;
  double _hx;
  
}; // AdvectionFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Inflow functor.
 */
class InflowFunctor
{

public:

  /**
   * \param[in,out] data is a scalar quantity array to add inflow to
   */
  InflowFunctor(FluidQuantity fq, double v, double x0, double y0, double x1, double y1) :
    _w(fq._w), _h(fq._h),
    _ox(fq._ox), _oy(fq._oy), _hx(fq._hx),
    data(fq.src() ),
    v(v)
  {
    ix0 = (int)(x0/_hx - _ox); ix0 = ix0>0  ? ix0 : 0;
    ix1 = (int)(x1/_hx - _ox); ix1 = ix1<_w ? ix1 : _w;

    iy0 = (int)(y0/_hx - _oy); iy0 = iy0>0  ? iy0 : 0;
    iy1 = (int)(y1/_hx - _oy); iy1 = iy1<_h ? iy1 : _h;
  };
  
  /* Sets fluid quantity inside the given rect to value `v' */
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    // a modifier : faire un parallel dispatch 2d avec exactement le bon nombre
    // d'itérations....
    
    int ix, iy;
    index2coord(index,ix,iy,_w,_h);

    if (ix >= ix0 and ix<ix1 and
	iy >= iy0 and iy<iy1 ) {
      if ( fabs(data(ix,iy)) < fabs(v) )
	data(ix,iy) = v;
    }
   
  } // end operator()

  int _w, _h;
  double _ox, _oy;
  double _hx;
  Array2d data;
  double v;
  int ix0, ix1;
  int iy0, iy1;
  
}; // class InflowFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * BuidRHS functor.
 *
 * Builds the pressure right hand side as the negative divergence.
 *
 */
class BuildRHSFunctor
{

public:

  /**
   * \param[in,out] data is a scalar quantity array to add inflow to
   */
  BuildRHSFunctor(FluidSolver fs) :
    _r(fs._r),
    _u(fs._u->_src),
    _v(fs._v->_src),
    scale(1.0/fs._hx),
    _w(fs._w),
    _h(fs._h)
  {};

  /* Sets fluid quantity inside the given rect to value `v' */
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int ix, iy;
    index2coord(index,x,y,_w,_h);

    _r(x,y) = -scale * (_u(x + 1, y    ) - _u(x, y) +
			_v(x    , y + 1) - _v(x, y) );
    
  } // operator()

  Array2d _r;
  Array2d _u, _v;
  double scale;
  int _w,_h;
  
} // class BuildRHSFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Apply pressure functor.
 *
 * Applies the computed pressure to the velocity field.
 *
 */
// class ApplyPressureFunctor
// {

// public:

//   /**
//    * \param[in,out] data is a scalar quantity array to add inflow to
//    */
//   ApplyPressureFunctor(FluidSolver fs, double timestep) :
//     _p(fs._p),
//     _u(fs._u._src),
//     _v(fs._v._src),
//     scale(timestep/(fs._density*fs._hx),
//     _w(fs._w),
//     _h(fs._h)
//   {};

//   /* Sets fluid quantity inside the given rect to value `v' */
//   KOKKOS_INLINE_FUNCTION
//   void operator() (const int& index) const
//   {

//     // this functor is supposed to be launched with _w*_h iterations
    
//     int ix, iy;
//     index2coord(index,x,y,_w,_h);

//     _u(x,     y    ) -= scale * _p(x,y);
//     _u(x + 1, y    ) += scale * _p(x,y);
//     _v(x,     y    ) -= scale * _p(x,y);
//     _v(x,     y + 1) += scale * _p(x,y);

//     // deal with extra borders (left and right),
//     // !!! beware u is sized _w+1,_h
//     // !!! beware v is sized _w  ,_h+1
//     if (x==0) {
//       _u(0,y ) = 0.0;
//       _u(_w,y) = 0.0;
//     }

//     if (y==0) {
//       _v(x,0 ) = 0.0;
//       _v(x,_h) = 0.0;
//     }
    
//   } // operator()

//   Array2d _p;
//   Array2d _u, _v;
//   double scale;
//   int _w,_h;
  
// } // class ApplyPressureFunctor

#endif // FLUID_FUNCTORS_H_
