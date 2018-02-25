#ifndef FLUID_QUANTITY_H_
#define FLUID_QUANTITY_H_

#include "kokkos_shared.h"

// some utilities

/* Length of vector (x, y) */
KOKKOS_INLINE_FUNCTION
double length(double x, double y) const {
    return sqrt(x*x + y*y);
}

/* Cubic pulse function.
 * Returns a value in range [0, 1].
 * Return value is 0 for x <= -1 and x >= 1; value is 1 for x=0
 * Smoothly interpolates between 0 and 1 between these three points.
 */
KOKKOS_INLINE_FUNCTION
double cubicPulse(double x) const {
    x = min(fabs(x), 1.0);
    return 1.0 - x*x*(3.0 - 2.0*x);
}

/* This is the class representing fluid quantities such as density and velocity
 * on the MAC grid. It saves attributes such as offset from the top left grid
 * cell, grid width and height as well as cell size.
 * 
 * It also contains two memory buffers: A source (_src) buffer and a
 * destination (_dst) buffer.
 * Most operations on fluid quantities can be done in-place; that is, they
 * write to the same buffer they're reading from (which is always _src).
 * However, some operations, such as advection, cannot be done in-place.
 * Instead, they will write to the _dst buffer. Once the operation is
 * completed, flip() can be called to swap the source and destination buffers,
 * such that the result of the operation is visible to subsequent operations.
 */
class FluidQuantity {

public:
  /* Memory buffers for fluid quantity */
  Array2d _src;
  Array2d _dst;
  
  /* Width and height */
  int _w;
  int _h;
  /* X and Y offset from top left grid cell.
   * This is (0.5,0.5) for centered quantities such as density,
   * and (0.0, 0.5) or (0.5, 0.0) for jittered quantities like the velocity.
   */
  double _ox;
  double _oy;
  /* Grid cell size */
  double _hx;

  /* Linear intERPolate between a and b for x ranging from 0 to 1 */
  KOKKOS_INLINE_FUNCTION
  double lerp(double a, double b, double x) const {
    return a*(1.0 - x) + b*x;
  }
  
public:
  FluidQuantity(int w, int h, double ox, double oy, double hx)
    : _w(w), _h(h), _ox(ox), _oy(oy), _hx(hx) {

    // kokkos view are initialized to zero
    _src = Array2d("_src",_w,_h);
    _dst = Array2d("_dst",_w,_h);

  }
    
  ~FluidQuantity() {
  }
    
  void flip() {

    Array2d tmp1(std::move(_src));
    Array2d tmp2(std::move(_dst));
    _src = tmp2;
    _dst = tmp1;
      
  }
  
  Array2d& src() {
      return _src;
  }

  /* Read-only and read-write access to grid cells */
  KOKKOS_INLINE_FUNCTION
  double at(int x, int y) const {
    return _src(x , y);
  }
  
  /* Linear intERPolate on grid at coordinates (x, y).
   * Coordinates will be clamped to lie in simulation domain
   */
  KOKKOS_INLINE_FUNCTION
  double lerp(double x, double y) const {
    
    x = fmin(fmax(x - _ox, 0.0), _w - 1.001);
    y = fmin(fmax(y - _oy, 0.0), _h - 1.001);
    int ix = (int)x;
    int iy = (int)y;
    x -= ix;
    y -= iy;
    
    double x00 = _src(ix + 0, iy + 0);
    double x10 = _src(ix + 1, iy + 0);
    double x01 = _src(ix + 0, iy + 1);
    double x11 = _src(ix + 1, iy + 1);
    
    return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y);
    
  } // lerp
    
}; // class FluidQuantity

#endif // FLUID_QUANTITY_H_
