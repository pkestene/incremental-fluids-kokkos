#ifndef FLUID_FUNCTORS_H_
#define FLUID_FUNCTORS_H_

#include "kokkos_shared.h"

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "FluidQuantity.h"
#include "SolidBody.h"

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
		   SolidBodyList bodies,
		   double timestep) :
    data(data),
    u(u),
    v(v),
    bodies(bodies),
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
		    SolidBodyList bodies,
		    double timestep) {
    const int size = data._w*data._h;
    AdvectionFunctor functor(data, u, v, bodies, timestep);
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
  
  /* Third order Runge-Kutta for velocity integration in time */
  KOKKOS_INLINE_FUNCTION
  void rungeKutta3(double &x,
		   double &y) const {
    
    double firstU = u.lerp(x, y)/_hx;
    double firstV = v.lerp(x, y)/_hx;
    
    double midX = x - 0.5*timestep*firstU;
    double midY = y - 0.5*timestep*firstV;
    
    double midU = u.lerp(midX, midY)/_hx;
    double midV = v.lerp(midX, midY)/_hx;
    
    double lastX = x - 0.75*timestep*midU;
    double lastY = y - 0.75*timestep*midV;
    
    double lastU = u.lerp(lastX, lastY);
    double lastV = v.lerp(lastX, lastY);
    
    x -= timestep*((2.0/9.0)*firstU + (3.0/9.0)*midU + (4.0/9.0)*lastU);
    y -= timestep*((2.0/9.0)*firstV + (3.0/9.0)*midV + (4.0/9.0)*lastV);
    
  } // rungeKutta3

  /* 
   * If the point (x, y) is inside a solid, project it back out to the
   * closest point on the surface of the solid.
   */
  KOKKOS_INLINE_FUNCTION
  void backProject(double &x, double &y) const
  {
    
    int rx = imin(imax((int)(x - _ox), 0), _w - 1);
    int ry = imin(imax((int)(y - _oy), 0), _h - 1);
    
    if (data._cell(rx,ry) != CELL_FLUID) {
      x = (x - _ox)*_hx;
      y = (y - _oy)*_hx;
      bodies(data._body(rx,ry)).closestSurfacePoint(x, y);
      x = x/_hx + _ox;
      y = y/_hx + _oy;
    }
  } // backProject
  
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
    rungeKutta3(x, y);

    /* If integrating back in time leaves us inside a solid
     * boundary (due to numerical error), make sure we
     * interpolate from a point inside the fluid.
     */
    backProject(x, y);

    // Second component: Interpolate from grid
    data._dst(ix,iy) = data.cerp(x, y);
    
  } // advection functor - operator()
  
  FluidQuantity data;
  FluidQuantity u;
  FluidQuantity v;
  SolidBodyList bodies;
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
 * Set fluid quantity inside the given rect to the specified value, but use
 * a smooth falloff to avoid oscillations
 *
 */
class InflowFunctor
{

public:

  /**
   * \param[in,out] data is a scalar quantity array to add inflow to
   */
  InflowFunctor(FluidQuantity& fq, double v, double x0, double y0, double x1, double y1) :
    x0(x0),
    y0(y0),
    x1(x1),
    y1(y1),
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
      double l = length(
			(2.0*(ix + 0.5)*_hx - (x0 + x1))/(x1 - x0),
			(2.0*(iy + 0.5)*_hx - (y0 + y1))/(y1 - y0)
			);
      double vi = cubicPulse(l)*v;
      if (fabs(data(ix,iy)) < fabs(vi))
	data(ix,iy) = vi;
      
    }
   
  } // end operator()

  double x0, y0, x1, y1;
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
   * \param[out]  r is the RHS   (_w  , _h  )
   * \param[in]   u is the x velocity (_w+1, _h  )
   * \param[in]   v is the y velocity (_w  , _h+1)
   * \param[in]   dVol d volume array
   * \param[in]   uVol u volume array
   * \param[in]   vVol v volume array
   * \param[in]   cell is the array of cell type (CELL_FLUID or CELL_SOLID)
   */
  BuildRHSFunctor(Array2d r, Array2d u, Array2d v,
		  Array2d dVol, Array2d uVol, Array2d vVol,
		  Array2d_uchar cell, double scale, int w, int h) :
    _r(r),
    _u(u),
    _v(v),
    _dVol(dVol),
    _uVol(uVol),
    _vVol(vVol),
    _cell(cell),
    _scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d r, Array2d u, Array2d v,
		    Array2d dVol, Array2d uVol, Array2d vVol,
		    Array2d_uchar cell,
		    double scale, int w, int h)
  {
    const int size = w*h;
    BuildRHSFunctor functor(r, u, v, dVol, uVol, vVol, cell, scale, w, h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    if (_cell(x,y) == CELL_FLUID) {
      _r(x,y) = -_scale *
	(_uVol(x+1,y) * _u(x+1,y) - _uVol(x,y)*_u(x, y) +
	 _vVol(x,y+1) * _v(x,y+1) - _vVol(x,y)*_v(x, y) );

      double vol = _dVol(x, y);
      
      //if (_bodies.empty())
      //continue;
      
    } else {
      _r(x,y) = 0.0;
    }
    
  } // operator()

  Array2d       _r;
  Array2d       _u, _v;
  Array2d       _dVol;
  Array2d       _uVol, _vVol;
  Array2d_uchar _cell;
  double        _scale;
  int           _w,_h;
  
}; // class BuildRHSFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * BuildPressureMatrix functor.
 *
 * Builds the pressure matrix, the following is the shape of
 * aDiag matrix; weights must be multiplied by "scale".
 *
 * -----------
 * |2|  3  |2|
 * -----------
 * |3|  4  |3|
 * -----------
 * |2|  3  |2|
 * -----------
 */
class BuildPressureMatrixFunctor
{

public:

  /**
   * \param[out] aDiag  (_w, _h)
   * \param[out] aPlusX (_w, _h)
   * \param[out] aPlusY (_w, _h)
   * \param[in]  cell (_w,_h) : array of cell type
   * \param[in]  uVol
   * \param[in]  vVol
   */
  BuildPressureMatrixFunctor(Array2d aDiag,
			     Array2d aPlusX,
			     Array2d aPlusY,
			     Array2d_uchar cell,
			     Array2d uVol,
			     Array2d vVol,
			     double scale,
			     int w, int h) :
    aDiag(aDiag),
    aPlusX(aPlusX),
    aPlusY(aPlusY),
    cell(cell),
    uVol(uVol),
    vVol(vVol),
    scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d aDiag,
		    Array2d aPlusX,
		    Array2d aPlusY,
		    Array2d_uchar cell,
		    Array2d uVol,
		    Array2d vVol,
		    double scale,
		    int w, int h)
  {
    const int size = w*h;
    BuildPressureMatrixFunctor functor(aDiag, aPlusX, aPlusY, cell,
				       uVol, vVol,
				       scale, w, h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    if (cell(x,y) == CELL_FLUID) {

      // for aDiag, we need to count the number of neighbors that are also
      // in state CELL_FLUID
      // default value is 4
      double tmp = 0;
      
      if ( x>0    and cell(x-1,y) == CELL_FLUID)
      	tmp += scale*uVol(x,y);
      if ( x<_w-1 and cell(x+1,y) == CELL_FLUID)
      	tmp += scale*uVol(x+1,y);
      if ( y>0    and cell(x,y-1) == CELL_FLUID)
      	tmp += scale*vVol(x,y);
      if ( y<_h-1 and cell(x,y+1) == CELL_FLUID)
      	tmp += scale*vVol(x,y+1);

      aDiag(x,y) = tmp;

      // compute aPlusX
      if ( x<_w-1 and cell(x+1,y) == CELL_FLUID)
	aPlusX(x,y) = scale*uVol(x+1,y);
      else
	aPlusX(x,y) = 0.0;

      // compute aPlusY
      if ( (y<_h-1 and cell(x,y+1) == CELL_FLUID) )
	aPlusY(x,y) = scale*vVol(x,y+1);
      else
	aPlusY(x,y) = 0.0;
      
    } else {
      // don't do anything,
      // aDiag, aPlusX, aPlusY have been reset previous calling this functor
    }
  } // operator()

  Array2d       aDiag;
  Array2d       aPlusX, aPlusY;
  Array2d_uchar cell;
  Array2d       uVol, vVol;
  double        scale;
  int           _w,_h;
  
}; // class BuildPressureMatrixFunctor


// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Apply preconditioner to vector `a' and store it in `dst' 
 *
 * Just use a SOR (Successive Over Relaxation) Gauss-Seidel based
 * preconditioner.
 *
 * https://en.wikipedia.org/wiki/Successive_over-relaxation
 *
 */
class ApplyPreconditionerFunctor
{

public:

  enum RedBlack_t {
    RED = 0,
    BLACK = 1
  };

  /**
   * \param[out] dst
   * \param[in]  a
   * \param[in]  cell : array of cell type
   * \param[in]  w width
   * \param[in]  h height
   * \param[in]  omega relaxation parameter
   * \param[in]  red_black type (can only be RED or BLACK)
   */
  ApplyPreconditionerFunctor(Array2d dst,
			     Array2d a,
			     Array2d_uchar cell,
			     int w,
			     int h,
			     double omega,
			     RedBlack_t red_black) :
    dst(dst),
    a(a),
    cell(cell),
    w(w),
    h(h),
    omega(omega),
    red_black(red_black),
    scale(1.0)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d dst, Array2d a, Array2d_uchar cell,
                    int w, int h,
		    double omega,
		    int nbIter)
  {
    // max size of a diagonal is min(w,h)
    const int size = w*h;
    for (int iter=0; iter<nbIter; ++iter) {

      // forward
      {
	ApplyPreconditionerFunctor functor(dst,a,cell,w,h,omega,RED);
	Kokkos::parallel_for(size, functor);
      }
      {
	ApplyPreconditionerFunctor functor(dst,a,cell,w,h,omega,BLACK);
	Kokkos::parallel_for(size, functor);
      }

      // backward (inverse colors)
      {
	ApplyPreconditionerFunctor functor(dst,a,cell,w,h,omega,BLACK);
	Kokkos::parallel_for(size, functor);
      }
      {
	ApplyPreconditionerFunctor functor(dst,a,cell,w,h,omega,RED);
	Kokkos::parallel_for(size, functor);
      }

    }
  }

  KOKKOS_INLINE_FUNCTION
  void do_red_black(int &x, int &y) const
  {
    
    double diag = 0.0, offDiag = 0.0;

    /* Here we build the matrix implicitly as the five-point
     * stencil. Grid borders are assumed to be solid, i.e.
     * there is no fluid outside the simulation domain.
     */
    if (cell(x,y)!=CELL_FLUID) {
      dst(x,y) =0.0;
      return;
    }
    
    if (x > 0     and cell(x-1,y)==CELL_FLUID) {
      diag    += scale;
      offDiag -= scale*dst(x - 1, y    );
    }
    if (y > 0     and cell(x,y-1)==CELL_FLUID) {
      diag    += scale;
      offDiag -= scale*dst(x    , y - 1);
    }
    if (x < w - 1 and cell(x+1,y)==CELL_FLUID) {
      diag    += scale;
      offDiag -= scale*dst(x + 1, y    );
    }
    if (y < h - 1 and cell(x,y+1)==CELL_FLUID) {
      diag    += scale;
      offDiag -= scale*dst(x    , y + 1);
    }
    
    // here is the Successive Over-Relaxation
    double new_val = (1-omega)*dst(x,y) + omega * ( a(x,y) - offDiag ) / diag;
    
    dst(x,y) = new_val;
    
  } // do_red_black

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,w,h);

    // if red_black == RED then compute only if x and y have different parity
    // if red_black == BLACK then compute only if x and y have same parity
    if ( !((x+y+red_black)&1) ) {
      do_red_black(x,y);
    }
    
  } // operator()

  Array2d dst ,a;
  Array2d_uchar cell;
  int w, h;
  double omega;
  RedBlack_t red_black;
  double scale;
  
}; // class ApplyPreconditionerFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Apply preconditioner to vector `a' and store it in `dst' 
 *
 * Just use a Jacobi preconditioner.
 *
 * https://en.wikipedia.org/wiki/
 *
 */
class ApplyJacobiPreconditionerFunctor
{

public:
  
  /**
   * \param[out] dst
   * \param[in]  a
   * \param[in]  cell : array of cell type
   * \param[in]  aDiag : array of diagonal terms of matrix A
   * \param[in]  w width
   * \param[in]  h height
   */
  ApplyJacobiPreconditionerFunctor(Array2d dst,
				   Array2d a,
				   Array2d_uchar cell,
				   Array2d aDiag,
				   int w,
				   int h) :
    dst(dst),
    a(a),
    cell(cell),
    aDiag(aDiag),
    w(w),
    h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d dst,
		    Array2d a,
		    Array2d_uchar cell,
		    Array2d aDiag,
                    int w, int h)
  {
    // max size of a diagonal is min(w,h)
    const int size = w*h;
    ApplyJacobiPreconditionerFunctor functor(dst,a,cell,aDiag,w,h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,w,h);

    if (cell(x,y) != CELL_FLUID)
      return;
    dst(x,y) = a(x,y) / aDiag(x,y);
     
  } // operator()

  Array2d dst ,a;
  Array2d_uchar cell;
  Array2d aDiag;
  int w, h;
  
}; // class ApplyJacobiPreconditionerFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Dot product functor.
 *
 *
 */
class DotProductFunctor
{

public:

  /**
   * \param[in]  a
   * \param[in]  b 
   * \param[out] r is result
   */
  DotProductFunctor(Array2d a, Array2d b) :
    a(a),
    b(b),
    w(a.dimension_0()),
    h(a.dimension_1())
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d a, Array2d b, double& result)
  {
    const int size = a.dimension_0()*a.dimension_1();
    DotProductFunctor functor(a, b);
    Kokkos::parallel_reduce(size, functor, result);
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (double& dst) const
  {
    // The identity under + is 0.
    dst = 0.0;
  } // init

  // this is were "action" / reduction takes place
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, double& result) const
  {

    int x, y;
    index2coord(index,x,y,w,h);

    result += a(x,y)*b(x,y);
        
  } // operator()
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile double& dst,
             const volatile double& src) const
  {
      dst += src;
  } // join

  Array2d a, b;
  int w,h;
  
}; // class DotProductFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Matrix vector product functor.
 *
 *
 */
class MatrixVectorProductFunctor
{

public:

  /**
   * \param[in,out]  dst
   * \param[in]  b
   */
  MatrixVectorProductFunctor(Array2d dst, Array2d b,
			     Array2d aDiag,
			     Array2d aPlusX, Array2d aPlusY) :
    dst(dst),
    b(b),
    aDiag(aDiag), aPlusX(aPlusX), aPlusY(aPlusY),
    w(dst.dimension_0()),
    h(dst.dimension_1())
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d dst, Array2d b,
		    Array2d aDiag,
		    Array2d aPlusX, Array2d aPlusY)
  {
    const int size = dst.dimension_0()*dst.dimension_1();
    MatrixVectorProductFunctor functor(dst, b, aDiag, aPlusX, aPlusY);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {    
    int x, y;
    index2coord(index,x,y,w,h);

    double t = aDiag(x,y)*b(x,y);
    
    if (x > 0)
      t += aPlusX(x-1,y  )*b(x-1,y  );
    if (y > 0)
      t += aPlusY(x  ,y-1)*b(x  ,y-1);
    if (x < w - 1)
      t += aPlusX(x  ,y  )*b(x+1,y  );
    if (y < h - 1)
      t += aPlusY(x  ,y  )*b(x  ,y+1);
    
    dst(x,y) = t;
  }

  Array2d dst, b;
  Array2d aDiag, aPlusX, aPlusY;
  int w,h;
  
}; // class MatrixVectorProductFunctor

  
// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Infinity Norm functor.
 *
 *
 */
class InfinityNormFunctor
{

public:

  /**
   * \param[in] a an Array2d
   */
  InfinityNormFunctor(Array2d a) :
    a(a),
    w(a.dimension_0()),
    h(a.dimension_1())
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d a, double& norm)
  {
    const int size = a.dimension_0()*a.dimension_1();
    InfinityNormFunctor functor(a);
    Kokkos::parallel_reduce(size, functor, norm);
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
  void operator() (const int& index, double& norm) const
  {

    int x, y;
    index2coord(index,x,y,w,h);
    norm = fmax(norm, fabs(a(x,y)));
        
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

  Array2d a;
  int w,h;
  
}; // class InfinityNormFunctor

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
   * \param[in]     cell is array of cell type (_w,_h)
   */
  ApplyPressureFunctor(Array2d p,
		       Array2d u, Array2d v,
		       Array2d_uchar cell,
		       double scale, int w, int h) :
    _p(p),
    _u(u),
    _v(v),
    _cell(cell),
    _scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d p,
		    Array2d u, Array2d v,
		    Array2d_uchar cell,
		    double scale, int w, int h)
  {
    const int size = w*h;
    ApplyPressureFunctor functor(p, u, v, cell, scale, w, h);
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

    if (_cell(x,y) == CELL_FLUID) {
      
      _u(x, y) -= _scale * _p(x  ,y  );
      if (x>0)
	_u(x, y) += _scale * _p(x-1,y  );
      
      _v(x, y) -= _scale * _p(x  ,y  );
      if (y>0)
	_v(x, y) += _scale * _p(x  ,y-1);

    }

  } // operator()

  Array2d _p;
  Array2d _u, _v;
  Array2d_uchar _cell;
  double _scale;
  int _w,_h;
  
}; // class ApplyPressureFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Set boundary condition functor.
 *
 * Sets all velocity cells bordering solid cells to the solid velocity.
 *
 */
class SetBoundaryConditionFunctor
{

public:

  /**
   * \param[in,out] u is the x velocity (_w+1, _h  )
   * \param[in,out] v is the y velocity (_w  , _h+1)
   * \param[in]     cell is array of cell type (_w,_h)
   */
  SetBoundaryConditionFunctor(Array2d u,
			      Array2d v,
			      Array2d_uchar cell,
			      Array2d_uchar body,
			      SolidBodyList bodies,
			      int w, int h, double hx) :
    _u(u),
    _v(v),
    _cell(cell),
    _body(body),
    _bodies(bodies),
    _w(w),
    _h(h),
    _hx(hx)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d u, Array2d v,
		    Array2d_uchar cell,
		    Array2d_uchar body,
		    SolidBodyList bodies,
		    int w, int h, double hx)
  {
    const int size = w*h;
    SetBoundaryConditionFunctor functor(u, v, cell, body, bodies, w, h, hx);
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

    if (_cell(x,y) == CELL_SOLID) {
      
      const SolidBody &b = _bodies(_body(x,y));
      
      _u(x    , y    ) = b.velocityX(  x       *_hx , (y + 0.5)*_hx );
      
    }

    if (x>0 and _cell(x-1,y) == CELL_SOLID) {

      const SolidBody &b = _bodies(_body(x-1,y));
      
      _u(x    , y    ) = b.velocityX(  x       *_hx , (y + 0.5)*_hx );
      
    }


    if (_cell(x,y) == CELL_SOLID) {

      const SolidBody &b = _bodies(_body(x,y));

      _v(x    , y    ) = b.velocityY( (x + 0.5)*_hx ,  y       *_hx );
      
    }

    if (y>0 and _cell(x,y-1) == CELL_SOLID) {

      const SolidBody &b = _bodies(_body(x,y-1));

      _v(x    , y    ) = b.velocityY( (x + 0.5)*_hx ,  y       *_hx );
      
    }

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

  Array2d _u, _v;
  Array2d_uchar _cell;
  Array2d_uchar _body;
  SolidBodyList _bodies;
  int _w,_h;
  double _hx;

}; // class SetBoundaryConditionFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * FillSolidFields functor.
 *
 * For a given fluid quantity, fill body and cell arrays.
 *
 *    body contains the index of the closest body to a given x,y location
 *    cell indicates if location x,y is FLUID or SOLID
 *
 */
class FillSolidFieldsFunctor {

public:

  /**
   * \param[in,out]     cell is array of cell type (_w,_h)
   * \param[in,out]     body is array of body Id per cell (_w,_h)
   * \param[in,out]     normalX
   * \param[in,out]     normalY
   * \param[in]         bodies
   */
  FillSolidFieldsFunctor(Array2d_uchar cell,
			 Array2d_uchar body,
			 Array2d       normalX,
			 Array2d       normalY,
			 SolidBodyList bodies,
			 int w,
			 int h,
			 double ox,
			 double oy,
			 double hx) :
    _cell(cell),
    _body(body),
    _normalX(normalX),
    _normalY(normalY),
    _bodies(bodies),
    _w(w),
    _h(h),
    _ox(ox),
    _oy(oy),
    _hx(hx)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d_uchar cell,
		    Array2d_uchar body,
		    Array2d       normalX,
		    Array2d       normalY,
		    SolidBodyList bodies,
		    int w,
		    int h,
		    double ox,
		    double oy,
		    double hx)
  {
    const int size = w*h;
    FillSolidFieldsFunctor functor(cell, body, normalX, normalY, bodies, w, h, ox, oy, hx);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    // this functor is supposed to be launched with _w*_h iterations
    // it has been slightly modified (compared to serial version)
    // to avoid data race
    
    int ix, iy;
    index2coord(index,ix,iy,_w,_h);

    double x = (ix + _ox)*_hx;
    double y = (iy + _oy)*_hx;
    
    // Search closest solid
    _body(ix,iy) = 0;
    double d = _bodies(0).distance(x, y);
    for (unsigned i = 1; i < _bodies.size(); i++) {
      double id = _bodies(i).distance(x, y);
      if (id < d) {
	_body(ix,iy) = i;
	d = id;
      }
    }
    
    /* 
     * If distance to closest solid is negative, this cell must be
     * inside it
     */
    if (d < 0.0)
      _cell(ix,iy) = CELL_SOLID;
    else
      _cell(ix,iy) = CELL_FLUID;

    /*
     * compute _normalX, _normalY
     */
    _bodies(_body(ix,iy)).distanceNormal(_normalX(ix,iy), _normalY(ix,iy), x, y);
    
  } // end operator()
  
  Array2d_uchar _cell;
  Array2d_uchar _body;
  SolidBodyList _bodies;
  Array2d       _normalX;
  Array2d       _normalY;
  int _w;
  int _h;
  double _ox;
  double _oy;
  double _hx;

}; // class FillSolidFieldsFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * FillSolidMask functor.
 *
 * Prepare auxiliary array for extrapolation.
 * The purpose of extrapolation is to extrapolate fluid quantities into
 * solids, where these quantities would normally be undefined. However, we
 * need these values for stable interpolation and boundary conditions.
 *
 * The way these are extrapolated here is by essentially solving a PDE,
 * such that the gradient of the fluid quantity is 0 along the gradient
 * of the distance field. This is essentially a more robust formulation of
 * "Set quantity inside solid to the value at the closest point on the
 * solid-fluid boundary"
 *
 * This PDE has a particular form which makes it very easy to solve exactly
 * using an upwinding scheme. What this means is that we can solve it from
 * outside-to-inside, with information flowing along the normal from the
 * boundary.
 *
 * Specifically, we can solve for the value inside a fluid cell using
 * extrapolateNormal if the two adjacent grid cells in "upstream" direction
 * (where the normal points to) are either fluid cells or have been solved
 * for already.
 *
 * The mask array keeps track of which cells wait for which values. If an
 * entry is 0, it means both neighbours are available and the cell is ready
 * for the PDE solve. If it is 1, the cell waits for the neighbour in
 * x direction, 2 for y-direction and 3 for both.
 *
 *
 * Here we implement a kokkos port of the refactored step4, found in 
 * https://github.com/pkestene/incremental-fluids in branch step4_refactoring, 
 * using directory 4-solid-boundaries_refactored.
 *
 * We changed the main data structure: mask is an unordered map, which in turn becomes
 * a Kokkos::UnorderedMap. We also restore some data parallelism.
 */
class FillSolidMaskFunctor {

public:

  // a custom data structure for the parallel_reduce sum
  struct MaskSumCell {
    int nbTodo;
    int nbReady;
  };
  
  /**
   * \param[in]         cell is array of cell types
   * \param[in,out]     mask is array of state of solid cell cells
   * \param[in]         normalX
   * \param[in]         normalY
   */
  FillSolidMaskFunctor(Array2d_uchar cell,
		       Array2d_uchar mask,
		       Array2d       normalX,
		       Array2d       normalY,
                       int w,
		       int h) :
    _cell(cell),
    _mask(mask),
    _normalX(normalX),
    _normalY(normalY),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d_uchar cell,
                    Array2d_uchar mask,
		    Array2d       normalX,
		    Array2d       normalY,
		    int w,
		    int h,
		    int& nbTodo,
		    int& nbReady)
  {
    MaskSumCell sumCell = {0, 0};
    const int size = w*h;
    FillSolidMaskFunctor functor(cell, mask,
				 normalX, normalY, w, h);
    Kokkos::parallel_reduce(size, functor, sumCell);
    nbTodo = sumCell.nbTodo;
    nbReady = sumCell.nbReady;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, MaskSumCell& sumCell) const
  {

    // this functor is supposed to be launched with _w*_h iterations
    // it has been slightly modified (compared to serial version)
    // to avoid data race
    
    int ix, iy;
    index2coord(index,ix,iy,_w,_h);

    if (_cell(ix,iy) != CELL_FLUID) {

      double nx = _normalX(ix,iy);
      double ny = _normalY(ix,iy);
      
      if ( (nx != 0.0 && _cell(ix + sgn(nx) , iy) != CELL_FLUID) ||
	   (ny != 0.0 && _cell(ix , iy + sgn(ny)) != CELL_FLUID) ) {
	_mask(ix,iy) = TODO;
	sumCell.nbTodo++;
      } else {
	_mask(ix,iy) = READY;
	sumCell.nbReady++;
      }
      
    }

  } // end operator()

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (MaskSumCell& dst) const
  {
    dst = {0, 0};
  } // init

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile MaskSumCell& dst,
             const volatile MaskSumCell& src) const
  {
    dst.nbTodo  += src.nbTodo;
    dst.nbReady += src.nbReady;
  } // join

  Array2d_uchar _cell;
  Array2d_uchar _mask;
  Array2d       _normalX;
  Array2d       _normalY;
  int _w;
  int _h;

}; // class FillSolidMaskFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Extrapolate inside solid cells functor.
 *
 * compute cells that are in READY mode right after fillSolidMask,
 * modify mask values from TODO to DONE.
 * 
 */
class ExtrapolateSolidCellReadyFunctor {
  
public:

  enum ExtrapolateStep_t {
    STEP1,
    STEP2
  };

  /**
   * \param[in,out]     src is an array to extrapolate inside solid
   * \param[in]         cell is array of cell types
   * \param[in,out]     mask is array of solide cell states
   * \param[in]         normalX
   * \param[in]         normalY
   * \param[in]         bodies
   */
  ExtrapolateSolidCellReadyFunctor(Array2d       src,
				   Array2d_uchar cell,
				   Array2d_uchar mask,
				   Array2d       normalX,
				   Array2d       normalY,
				   int           w,
				   int           h,
				   int           step) :
    _src(src),
    _cell(cell),
    _mask(mask),
    _normalX(normalX),
    _normalY(normalY),
    _w(w),
    _h(h),
    _step(step)
  {};

  // static method which does it all: create and execute functor
  static int apply(Array2d       src,
		   Array2d_uchar cell,
		   Array2d_uchar mask,
		   Array2d       normalX,
		   Array2d       normalY,
		   int           w,
		   int           h,
		   int           step)
  {
    // this number is used to accumulate the number of cell that flips
    // state to DONE (in step2).
    // in step 1, just return 0 (non relevant)
    int nbDoneFlip = 0;
    const int size = w*h;
    ExtrapolateSolidCellReadyFunctor functor(src, cell, mask,
					     normalX, normalY, w, h,step);
    Kokkos::parallel_reduce(size, functor, nbDoneFlip);
    return nbDoneFlip;
  }

  /* Solve for value at index idx using values of neighbours in normal x/y
   * direction. The value is computed such that the directional derivative
   * along distance field normal is 0.
   */
  KOKKOS_INLINE_FUNCTION
  double extrapolateNormal(int ix, int iy) const {
    double nx = _normalX(ix,iy);
    double ny = _normalY(ix,iy);
        
    double srcX = _src(ix + sgn(nx),iy         );
    double srcY = _src(ix          ,iy+ sgn(ny));
        
    return (fabs(nx)*srcX + fabs(ny)*srcY)/(fabs(nx) + fabs(ny));
  } // extrapolateNormal
  
  /* Given that a neighbour in upstream direction specified by mask (1=x, 2=y)
   * now has been solved for, update the mask appropriately and, if this cell
   * can now be computed, add it to the queue of ready cells
   */
  // return true if at least 2 neighbors are in DONE state,
  // meaning all upwind neighbors have already been computed,
  // and meaning that current cell can flip in READY
  KOKKOS_INLINE_FUNCTION
  bool is_ready_to_compute(int ix, int iy) const {

    int nbDone=0;
    if (_cell(ix-1,iy) != CELL_FLUID and _mask(ix-1,iy) == DONE)
      nbDone++;
    if (_cell(ix+1,iy) != CELL_FLUID and _mask(ix+1,iy) == DONE)
      nbDone++;
    if (_cell(ix,iy-1) != CELL_FLUID and _mask(ix,iy-1) == DONE)
      nbDone++;
    if (_cell(ix,iy+1) != CELL_FLUID and _mask(ix,iy+1) == DONE)
      nbDone++;
    
    return (nbDone >=2);
    
  } // is_ready_to_compute

  KOKKOS_INLINE_FUNCTION
  void step1 (const int& ix, const int& iy, int& count) const
  {
    
    if (_cell(ix,iy) != CELL_FLUID and _mask(ix,iy) == READY) {
      
      // cell can be computed
      // Solve for value in cell */
      _src(ix,iy) = extrapolateNormal(ix,iy);
      
      //_mask_map.insert(idx,DONE);
      _mask(ix,iy) = DONE;
      
    }

  } // end step1
  
  KOKKOS_INLINE_FUNCTION
  void step2 (const int& ix, const int& iy, int& count) const
  {

    if (_cell(ix,iy) != CELL_FLUID and
	_mask(ix,iy) != DONE and
	is_ready_to_compute(ix,iy)) {
      
      // cell can be computed
      // Solve for value in cell */
      _src(ix,iy) = extrapolateNormal(ix,iy);
      
      //_mask_map.insert(idx,DONE);
      _mask(ix,iy) = DONE;

      count++;
    }
    
  } // end step2
  
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, int& count) const
  {

    int ix, iy;
    index2coord(index,ix,iy,_w,_h);

    // if then else step1 / step2 
    if (_step == STEP1) {

      step1(ix,iy,count);

    } else if (_step == STEP2) {

      step2(ix,iy,count);

    }
    
  } // end operator()
  
  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (int& dst) const
  {
    dst = 0;
  } // init

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile       int& dst,
             const volatile int& src) const
  {
    dst  += src;
  } // join

  Array2d       _src;
  Array2d_uchar _cell;
  Array2d_uchar _mask;
  Array2d       _normalX;
  Array2d       _normalY;
  int           _w;
  int           _h;
  int           _step;
  
};  // class ExtrapolateSolidCellReadyFunctor

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Get number of cells such that value is in a given state functor.
 *
 * compute number cells in a given state.
 * 
 */
class GetNumberofCellsPerStateFunctor {

public:
  /**
   * \param[in,out]     src is an array to extrapolate inside solid
   * \param[in]         cell is array of cell types
   * \param[in,out]     mask_map is array of
   * \param[in]         normalX
   * \param[in]         normalY
   * \param[in]         bodies
   */
  GetNumberofCellsPerStateFunctor(Array2d_uchar mask,
				  int           w,
				  int           h,
				  uint8_t       state) : 
    _mask(mask),
    _w(w),
    _h(h),
    _state(state)
  {};

  // static method which does it all: create and execute functor
  static int apply(Array2d_uchar mask,
		   int           w,
		   int           h,
		   uint8_t       state)
  {
    const int size = w*h;
    int nbState = 0;
    GetNumberofCellsPerStateFunctor functor(mask, w, h, state);
    Kokkos::parallel_reduce(size, functor, nbState);
    return nbState;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index, int& count) const
  {

    int ix, iy;
    index2coord(index,ix,iy,_w,_h);

    if (_mask(ix,iy) == _state) {

      count++;
      
    }

  } // end operator()

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (int& dst) const
  {
    dst = 0;
  } // init

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile       int& dst,
             const volatile int& src) const
  {
    dst  += src;
  } // join

  Array2d_uchar _mask;
  int           _w;
  int           _h;
  uint8_t       _state;

};  // class GetNumberofCellsPerStateFunctor

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


#endif // FLUID_FUNCTORS_H_
