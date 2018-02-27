#ifndef FLUID_FUNCTORS_H_
#define FLUID_FUNCTORS_H_

#include "kokkos_shared.h"

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "FluidQuantity.h"

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

  // static method which does it all: create and execute functor
  static void apply(FluidQuantity data,
		    FluidQuantity u,
		    FluidQuantity v,
		    double timestep) {
    const int size = data._w*data._h;
    AdvectionFunctor functor(data, u, v, timestep);
    Kokkos::parallel_for(size, functor);
  }
    
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
 *
 * Sets fluid quantity inside the given rect to value `v'.
 *
 */
class InflowFunctor
{

public:

  /**
   * \param[in,out] data is a scalar quantity array to add inflow to
   */
  InflowFunctor(FluidQuantity& fq, double v, double x0, double y0, double x1, double y1) :
    _w(fq._w),
    _h(fq._h),
    _ox(fq._ox),
    _oy(fq._oy),
    _hx(fq._hx),
    data( fq.src() ),
    v(v)
  {
    ix0 = (int)(x0/_hx - _ox); ix0 = ix0>0  ? ix0 : 0;
    ix1 = (int)(x1/_hx - _ox); ix1 = ix1<_w ? ix1 : _w;

    iy0 = (int)(y0/_hx - _oy); iy0 = iy0>0  ? iy0 : 0;
    iy1 = (int)(y1/_hx - _oy); iy1 = iy1<_h ? iy1 : _h;

  };

  // static method which does it all: create and execute functor
  static void apply(FluidQuantity& fq,
		    double v,
		    double x0, double y0, double x1, double y1)
  {
    const int size = fq._w*fq._h;
    InflowFunctor functor(fq, v, x0, y0, x1, y1);
    Kokkos::parallel_for(size, functor);
  }
    
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    // TODO : modify in favour of a 2d dispatch with exactely the right number 
    // of iterations.
    
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
   * \param[in]     r is the RHS   (_w  , _h  )
   * \param[in,out] u is the x velocity (_w+1, _h  )
   * \param[in,out] v is the y velocity (_w  , _h+1)
   */
  BuildRHSFunctor(Array2d r, Array2d u, Array2d v, double scale, int w, int h) :
    _r(r),
    _u(u),
    _v(v),
    scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d r, Array2d u, Array2d v, double scale, int w, int h)
  {
    const int size = w*h;
    BuildRHSFunctor functor(r, u, v, scale, w, h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    _r(x,y) = -scale * (_u(x + 1, y    ) - _u(x, y) +
			_v(x    , y + 1) - _v(x, y) );
    
  } // operator()

  Array2d _r;
  Array2d _u, _v;
  double scale;
  int _w,_h;
  
}; // class BuildRHSFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Apply pressure functor.
 *
 * Applies the computed pressure to the velocity field.
 *
 */
class ApplyPressureFunctor
{

public:

  /**
   * \param[in]     p is the pressure   (_w  , _h  )
   * \param[in,out] u is the x velocity (_w+1, _h  )
   * \param[in,out] v is the y velocity (_w  , _h+1)
   */
  ApplyPressureFunctor(Array2d p,
		       Array2d u, Array2d v,
		       double scale, int w, int h) :
    _p(p),
    _u(u),
    _v(v),
    _scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d p,
		    Array2d u, Array2d v,
		    double scale, int w, int h)
  {
    const int size = w*h;
    ApplyPressureFunctor functor(p, u, v, scale, w, h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    // this functor is supposed to be launched with _w*_h iterations
    // it has been slightly modified (compared to serial version)
    // to avoid data race
    
    int x, y;
    index2coord(index,x,y,_w,_h);

    _u(x, y) -= _scale * _p(x  ,y  );
    if (x>0)
      _u(x, y) += _scale * _p(x-1,y  );
    
    _v(x, y) -= _scale * _p(x  ,y  );
    if (y>0)
      _v(x, y) += _scale * _p(x  ,y-1);

    // deal with extra borders (left and right),
    // !!! beware u is sized _w+1,_h
    // !!! beware v is sized _w  ,_h+1
    if (x==0) {
      _u(0 , y ) = 0.0;
      _u(_w, y ) = 0.0;
    }

    if (y==0) {
      _v(x , 0 ) = 0.0;
      _v(x , _h) = 0.0;
    }
    
  } // operator()

  Array2d _p;
  Array2d _u, _v;
  double _scale;
  int _w,_h;
  
}; // class ApplyPressureFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * MaxVelocity functor.
 *
 *  Returns the maximum allowed timestep. Note that the actual timestep
 *  taken should usually be much below this to ensure accurate
 *  simulation - just never above.
 *
 */
class MaxVelocityFunctor
{

public:

  /**
   * \param[in]     r is the RHS   (_w  , _h  )
   * \param[in,out] u is the x velocity (_w+1, _h  )
   * \param[in,out] v is the y velocity (_w  , _h+1)
   */
  MaxVelocityFunctor(FluidQuantity u, FluidQuantity v, int w, int h) :
    _u(u),
    _v(v),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(FluidQuantity u, FluidQuantity v, double& maxVelocity,
		    int w, int h)
  {
    const int size = w*h;
    MaxVelocityFunctor functor(u, v, w, h);
    Kokkos::parallel_reduce(size, functor, maxVelocity);
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (double& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<double>::min();
#endif // __CUDA_ARCH__
  } // init

  // this is were "action" / reduction takes place
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, double& maxVelocity) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    /* Average velocity at grid cell center */
    double u = _u.lerp(x + 0.5, y + 0.5);
    double v = _v.lerp(x + 0.5, y + 0.5);
    
    double velocity = sqrt(u*u + v*v);
    maxVelocity = fmax(maxVelocity, velocity);
        
  } // operator()
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile double& dst,
             const volatile double& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  FluidQuantity _u, _v;
  int _w,_h;
  
}; // class MaxVelocityFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * \class ProjectFunctor_GaussSeidel
 * 
 * Performs the pressure solve using Gauss-Seidel iterations.
 * Please notice, this is not really Gauss-Seidel because of the naive 
 * parallelization; the results are slightly different from the serial
 * version; it just pragmatically "works". See below, the Jacobi iterations
 * which gives the same results in serial and in parallel.
 *
 * The solver will run as long as it takes to get the relative error below
 * a threshold, but will never exceed 'limit' iterations
 */
class ProjectFunctor_GaussSeidel
{

public:

  enum RedBlack_t {
    RED,
    BLACK
  };
  
  /**
   * \param[in,out] p is the pressure (_w  , _h  )
   * \param[in]     r is the RHS      (_w  , _h  )
   */
  ProjectFunctor_GaussSeidel(Array2d p, Array2d r, double scale, int w, int h,
			     RedBlack_t redblack_type) :
    _p(p),
    _r(r),
    _scale(scale),
    _w(w),
    _h(h),
    redblack_type(redblack_type)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d p, Array2d r, double scale,
		    double& maxDelta,
		    int w, int h,
		    RedBlack_t redblack_type)
  {
    const int size = w*h;
    ProjectFunctor_GaussSeidel functor(p, r, scale, w, h, redblack_type);
    Kokkos::parallel_reduce(size, functor, maxDelta);
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (double& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<double>::min();
#endif // __CUDA_ARCH__
  } // init

  KOKKOS_INLINE_FUNCTION
  void do_red_black(int &x, int &y,
		    double &maxDelta) const
  {
    
    double diag = 0.0, offDiag = 0.0;

    /* Here we build the matrix implicitly as the five-point
     * stencil. Grid borders are assumed to be solid, i.e.
     * there is no fluid outside the simulation domain.
     */
    if (x > 0) {
      diag    += _scale;
      offDiag -= _scale*_p(x - 1, y    );
    }
    if (y > 0) {
      diag    += _scale;
      offDiag -= _scale*_p(x    , y - 1);
    }
    if (x < _w - 1) {
      diag    += _scale;
      offDiag -= _scale*_p(x + 1, y    );
    }
    if (y < _h - 1) {
      diag    += _scale;
      offDiag -= _scale*_p(x    , y + 1);
    }

    double newP = ( _r(x,y) - offDiag ) / diag;
    maxDelta = fmax(maxDelta, fabs(_p(x,y) - newP));
    
    _p(x,y) = newP;
    
  } // compute_diag_offdiag
  
  
  // this is were "action" / reduction takes place
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, double& maxDelta) const
  {
    
    int x, y;
    index2coord(index,x,y,_w,_h);

    if (redblack_type == RED) { // x and y have same parity

      if ( ((x&1) and (y&1)) || (!(x&1) and !(y&1)) ) {

	do_red_black(x,y, maxDelta);

      }
      
    } else if (redblack_type == BLACK) { // x and y have different parity

      if ( (!(x&1) and (y&1)) || ((x&1) and !(y&1)) ) {

	do_red_black(x,y, maxDelta);

      }
      
    }
    
  } // operator()
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile double& dst,
             const volatile double& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  Array2d _p, _r;
  double  _scale;
  int     _w,_h;
  RedBlack_t redblack_type;
  
}; // class ProjectFunctor_GaussSeidel

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * \class ProjectFunctor_Jacobi
 * 
 * Performs the pressure solve using Jacobi iterations.
 * The solver will run as long as it takes to get the relative error below
 * a threshold, but will never exceed 'limit' iterations
 */
class ProjectFunctor_Jacobi
{

public:

  /**
   * \param[in,out] p is the pressure (_w  , _h  )
   * \param[in]     r is the RHS      (_w  , _h  )
   */
  ProjectFunctor_Jacobi(Array2d p, Array2d p2, Array2d r, double scale, int w, int h) :
    _p(p),
    _p2(p2),
    _r(r),
    _scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d p, Array2d p2, Array2d r, double scale,
		    double& maxDelta,
		    int w, int h)
  {
    const int size = w*h;
    ProjectFunctor_Jacobi functor(p, p2, r, scale, w, h);
    Kokkos::parallel_reduce(size, functor, maxDelta);
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (double& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<double>::min();
#endif // __CUDA_ARCH__
  } // init

  // this is were "action" / reduction takes place
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, double& maxDelta) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    double diag = 0.0, offDiag = 0.0;
    
    /* Here we build the matrix implicitly as the five-point
     * stencil. Grid borders are assumed to be solid, i.e.
     * there is no fluid outside the simulation domain.
     */
    if (x > 0) {
      diag    += _scale;
      offDiag -= _scale*_p(x - 1, y    );
    }
    if (y > 0) {
      diag    += _scale;
      offDiag -= _scale*_p(x    , y - 1);
    }
    if (x < _w - 1) {
      diag    += _scale;
      offDiag -= _scale*_p(x + 1, y    );
    }
    if (y < _h - 1) {
      diag    += _scale;
      offDiag -= _scale*_p(x    , y + 1);
    }
        
    double newP = ( _r(x,y) - offDiag ) / diag;

    maxDelta = fmax(maxDelta, fabs(_p(x,y) - newP));

    _p2(x,y) = newP;

  } // operator()
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile double& dst,
             const volatile double& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  Array2d _p, _p2, _r;
  double _scale;
  int _w,_h;
  
}; // class ProjectFunctor_Jacobi

#endif // FLUID_FUNCTORS_H_
