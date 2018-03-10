#ifndef FLUID_QUANTITY_H_
#define FLUID_QUANTITY_H_

#include "kokkos_shared.h"

#include "math_utils.h"

#include "SolidBody.h"

// mask type for SolidBody extrapolate computation
using MaskMap2d = Kokkos::UnorderedMap<int, uint8_t, Device>;

/* Cubic pulse function.
 * Returns a value in range [0, 1].
 * Return value is 0 for x <= -1 and x >= 1; value is 1 for x=0
 * Smoothly interpolates between 0 and 1 between these three points.
 */
KOKKOS_INLINE_FUNCTION
double cubicPulse(double x) {
  x = fmin(fabs(x), 1.0);
  return 1.0 - x*x*(3.0 - 2.0*x);
}


/* Enum to differentiate fluid and solid cells */
enum CellType {
    CELL_FLUID,
    CELL_SOLID
};

enum CellMaskState : uint8_t {
  TODO,
  READY,
  DONE
};

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
  
  /* Normal of distance field at grid points */
  Array2d _normalX;
  Array2d _normalY;

  /* Designates cells as fluid or solid cells (CELL_FLUID or CELL_SOLID) */
  Array2d_uchar _cell;

  /* Specifies the index of the rigid body closes to a grid cell */
  Array2d_uchar _body;

  /* Auxiliary array used for extrapolation */
  Array2d_uchar _mask;
  
  MaskMap2d _mask_map;
  
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

  /* Cubic intERPolate using samples a through d for x ranging from 0 to 1.
   * A Catmull-Rom spline is used. Over- and undershoots are clamped to
   * prevent blow-up.
   */
  KOKKOS_INLINE_FUNCTION
  double cerp(double a, double b, double c, double d, double x) const {

    double xsq = x*x;
    double xcu = xsq*x;
    
    double minV = fmin(a, fmin(b, fmin(c, d)));
    double maxV = fmax(a, fmax(b, fmax(c, d)));
    
    double t =
      a*(0.0 - 0.5*x + 1.0*xsq - 0.5*xcu) +
      b*(1.0 + 0.0*x - 2.5*xsq + 1.5*xcu) +
      c*(0.0 + 0.5*x + 2.0*xsq - 1.5*xcu) +
      d*(0.0 + 0.0*x - 0.5*xsq + 0.5*xcu);
    
    return fmin(fmax(t, minV), maxV);
    
  } // cerp

public:
  FluidQuantity(int w, int h, double ox, double oy, double hx)
    : _w(w), _h(h), _ox(ox), _oy(oy), _hx(hx) {
    
    // kokkos view are initialized to zero
    _src = Array2d("_src",_w,_h);
    _dst = Array2d("_dst",_w,_h);

    _normalX = Array2d("_normalX",_w,_h);
    _normalY = Array2d("_normalY",_w,_h);

    _cell = Array2d_uchar("_cell",_w,_h);
    _body = Array2d_uchar("_body",_w,_h);
    
    _mask_map = MaskMap2d(_w*_h);
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

  /* Cubic intERPolate on grid at coordinates (x, y).
   * Coordinates will be clamped to lie in simulation domain
   */
  KOKKOS_INLINE_FUNCTION
  double cerp(double x, double y) const {

    x = fmin(fmax(x - _ox, 0.0), _w - 1.001);
    y = fmin(fmax(y - _oy, 0.0), _h - 1.001);
    int ix = (int)x;
    int iy = (int)y;
    x -= ix;
    y -= iy;
    
    int x0 = imax(ix - 1, 0), x1 = ix, x2 = ix + 1, x3 = imin(ix + 2, _w - 1);
    int y0 = imax(iy - 1, 0), y1 = iy, y2 = iy + 1, y3 = imin(iy + 2, _h - 1);
    
    double q0 = cerp(at(x0, y0), at(x1, y0), at(x2, y0), at(x3, y0), x);
    double q1 = cerp(at(x0, y1), at(x1, y1), at(x2, y1), at(x3, y1), x);
    double q2 = cerp(at(x0, y2), at(x1, y2), at(x2, y2), at(x3, y2), x);
    double q3 = cerp(at(x0, y3), at(x1, y3), at(x2, y3), at(x3, y3), x);
    
    return cerp(q0, q1, q2, q3, y);
  }
  
}; // class FluidQuantity

#endif // FLUID_QUANTITY_H_
