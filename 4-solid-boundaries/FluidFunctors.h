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
    
    // Second component: Interpolate from grid
    data._dst(ix,iy) = data.cerp(x, y);
    
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
   * \param[in]     r is the RHS   (_w  , _h  )
   * \param[in,out] u is the x velocity (_w+1, _h  )
   * \param[in,out] v is the y velocity (_w  , _h+1)
   * \param[in]     cell is the array of cell type (CELL_FLUID or CELL_SOLID)
   */
  BuildRHSFunctor(Array2d r, Array2d u, Array2d v, Array2d_uchar cell, double scale, int w, int h) :
    _r(r),
    _u(u),
    _v(v),
    _cell(cell),
    _scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d r, Array2d u, Array2d v, Array2d_uchar cell,
		    double scale, int w, int h)
  {
    const int size = w*h;
    BuildRHSFunctor functor(r, u, v, cell, scale, w, h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    if (_cell(x,y) == CELL_FLUID) {
      _r(x,y) = -_scale * (_u(x + 1, y    ) - _u(x, y) +
			   _v(x    , y + 1) - _v(x, y) );
    } else {
      _r(x,y) = 0.0;
    }
    
  } // operator()

  Array2d       _r;
  Array2d       _u, _v;
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
   */
  BuildPressureMatrixFunctor(Array2d aDiag, Array2d aPlusX, Array2d aPlusY,
			     Array2d_uchar cell,
			     double scale, int w, int h) :
    aDiag(aDiag),
    aPlusX(aPlusX),
    aPlusY(aPlusY),
    cell(cell),
    scale(scale),
    _w(w),
    _h(h)
  {};

  // static method which does it all: create and execute functor
  static void apply(Array2d aDiag, Array2d aPlusX, Array2d aPlusY,
		    Array2d_uchar cell,
		    double scale, int w, int h)
  {
    const int size = w*h;
    BuildPressureMatrixFunctor functor(aDiag, aPlusX, aPlusY, cell, scale, w, h);
    Kokkos::parallel_for(size, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& index) const
  {

    int x, y;
    index2coord(index,x,y,_w,_h);

    if (cell(x,y) == CELL_FLUID) {
    
      if ( (x==0    and y==0   ) ||
	   (x==0    and y==_h-1) ||
	   (x==_w-1 and y==0   ) ||
	   (x==_w-1 and y==_h-1) ) { // corners
	aDiag(x,y) = 2*scale;
      } else if ( (x==0               and y>0     and y<_h-1) ||
		  (x==_w-1            and y>0     and y<_h-1) ||
		  (x>0     and x<_w-1 and y==0)               ||
		  (x>0     and x<_w-1 and y==_h-1) ) { // borders
	aDiag(x,y) = 3*scale;
      } else { // bulk
	aDiag(x,y) = 4*scale;
      }
      
      if (x<_w-1) {
	aPlusX(x,y) = -scale;
      } else {
	aPlusX(x,y) = 0.0;
      }
      
      if (y<_h-1) {
	aPlusY(x,y) = -scale;
      } else {
	aPlusY(x,y) = 0.0;
      }

      // take care of neighbor solid cells
      if (cell(x+1,y) == CELL_SOLID) {
	aDiag(x,y) -= scale;
	aPlusX(x,y) = 0.0;;
      }

      // take care of neighbor solid cells
      if (cell(x,y+1) == CELL_SOLID) {
	aDiag(x,y) -= scale;
	aPlusY(x,y) = 0.0;;
      }
      
    } else {
      // don't do anything,
      // aDiag, aPlusX, aPlusY have been reset previous calling this functor
    }
  } // operator()

  Array2d       aDiag;
  Array2d       aPlusX, aPlusY;
  Array2d_uchar cell;
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
    if (cell(x,y)==CELL_FLUID) {
      
      if (x > 0 and cell(x-1,y)==CELL_FLUID) {
	diag    += scale;
	offDiag -= scale*dst(x - 1, y    );
      }
      if (y > 0 and cell(x,y-1)==CELL_FLUID) {
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
      
    }
    
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
      _v(x    , y    ) = b.velocityY( (x + 0.5)*_hx ,  y       *_hx );
      _u(x + 1, y    ) = b.velocityX( (x + 1.0)*_hx , (y + 0.5)*_hx );
      _v(x    , y + 1) = b.velocityY( (x + 0.5)*_hx , (y + 1.0)*_hx );
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
