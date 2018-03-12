#ifndef SOLID_BODIES_H_
#define SOLID_BODIES_H_

#include <math.h>
#include "math_utils.h"

/* Rotates point (x, y) by angle phi */
KOKKOS_INLINE_FUNCTION
void rotate(double &x, double &y, double phi) {

  double tmpX = x, tmpY = y;
  x =  cos(phi)*tmpX + sin(phi)*tmpY;
  y = -sin(phi)*tmpX + cos(phi)*tmpY;
  
} // rotate

/* The base class representing solid bodies in the simulation.
 * It holds information about position, scale and rotation of
 * the solid as well as lateral and angular velocity.
 *
 * It does not represent any shape; this is handled by the subclasses.
 * To expose the shape to the simulation, methods for evaluating the
 * signed distance to the solid, the gradient of the distance function
 * and the closest point on the surface of the solid are exposed.
 */
class SolidBody {

public:
  enum solid_type {
    BOX,
    SPHERE
  };
  
public:
  solid_type _type;
  double _posX; /* Position */
  double _posY;
  double _scaleX; /* Scale */
  double _scaleY;
  double _theta; /* Rotation */
    
  double _velX; /* Lateral velocity */
  double _velY;
  double _velTheta; /* Angular velocity */
    
  /* Transforms point (x, y) form the global to the local coordinate system */
  KOKKOS_INLINE_FUNCTION
  void globalToLocal(double &x, double &y) const {
    x -= _posX;
    y -= _posY;
    rotate(x, y, -_theta);
    x /= _scaleX;
    y /= _scaleY;
  } // globalToLocal
  
  /* Transforms point (x, y) form the local to the global coordinate system */
  KOKKOS_INLINE_FUNCTION
  void localToGlobal(double &x, double &y) const {
    x *= _scaleX;
    y *= _scaleY;
    rotate(x, y, _theta);
    x += _posX;
    y += _posY;
  } // localToGlobal
    
public:

  KOKKOS_INLINE_FUNCTION
  SolidBody() {};
  
  KOKKOS_INLINE_FUNCTION
  SolidBody(solid_type type,
	    double posX, double posY, double scaleX, double scaleY,
	    double theta, double velX, double velY, double velTheta) :
    _type(type),
    _posX(posX), _posY(posY), _scaleX(scaleX), _scaleY(scaleY),
    _theta(theta), _velX(velX), _velY(velY), _velTheta(velTheta) {}

  KOKKOS_INLINE_FUNCTION
  ~SolidBody() {};
    
  /* Returns the signed distance from (x, y) to the nearest point on surface
   * of the solid. The distance is negative if (x, y) is inside the solid
   */
  KOKKOS_INLINE_FUNCTION
  double distance(double x, double y) const {
    if (_type == BOX)
      return distance_box(x,y);
    if (_type == SPHERE)
      return distance_sphere(x,y);
    return 0.0;
  } // distance

  KOKKOS_INLINE_FUNCTION
  double distance_box(double x, double y) const {
    x -= _posX;
    y -= _posY;
    rotate(x, y, -_theta);
    double dx = fabs(x) - _scaleX*0.5;
    double dy = fabs(y) - _scaleY*0.5;

    if (dx >= 0.0 || dy >= 0.0)
      return length(fmax(dx, 0.0), fmax(dy, 0.0));
    else
      return fmax(dx, dy);
  } // distance_box

  KOKKOS_INLINE_FUNCTION
  double distance_sphere(double x, double y) const {
    return length(x - _posX, y - _posY) - _scaleX*0.5;
  } // distance_sphere

  /* Changes (x, y) to lie on the closest point on the surface of the solid */
  KOKKOS_INLINE_FUNCTION
  void closestSurfacePoint(double &x, double &y) const {
    if (_type == BOX)
      closestSurfacePoint_box(x,y);
    if (_type == SPHERE)
      closestSurfacePoint_sphere(x,y);
  } // closestSurfacePoint

  KOKKOS_INLINE_FUNCTION
  void closestSurfacePoint_box(double &x, double &y) const {
    
    x -= _posX;
    y -= _posY;
    rotate(x, y, -_theta);
    double dx = fabs(x) - _scaleX*0.5;
    double dy = fabs(y) - _scaleY*0.5;
    
    if (dx > dy)
      x = nsgn(x)*0.5*_scaleX;
    else
      y = nsgn(y)*0.5*_scaleY;

    rotate(x, y, _theta);
    x += _posX;
    y += _posY;
    
  } // closestSurfacePoint_box

  KOKKOS_INLINE_FUNCTION
  void closestSurfacePoint_sphere(double &x, double &y) const {
    globalToLocal(x, y);
        
    double r = length(x, y);
    if (r < 1e-4) {
      x = 0.5;
      y = 0.0;
    } else {
      x /= 2.0*r;
      y /= 2.0*r;
    }
        
    localToGlobal(x, y);
  } // closestSurfacePoint_sphere
  
  KOKKOS_INLINE_FUNCTION
  void distanceNormal_box(double &nx, double &ny, double x, double y) const {
    x -= _posX;
    y -= _posY;
    rotate(x, y, -_theta);
    if (fabs(x) - _scaleX*0.5 > fabs(y) - _scaleY*0.5) {
      nx = nsgn(x);
      ny = 0.0;
    } else {
      nx = 0.0;
      ny = nsgn(y);
    }
    rotate(nx, ny, _theta);
    
  } // distanceNormal_box

  KOKKOS_INLINE_FUNCTION
  void distanceNormal_sphere(double &nx, double &ny, double x, double y) const {
    x -= _posX;
    y -= _posY;
    float r = length(x, y);
    if (r < 1e-4) {
      nx = 1.0;
      ny = 0.0;
    } else {
      nx = x/r;
      ny = y/r;
    }
  } // distanceNormal_sphere

  /* Returns the gradient of the distance function at (x, y) in (nx, ny) */
  KOKKOS_INLINE_FUNCTION
  void distanceNormal(double &nx, double &ny, double x, double y) const {
    if (_type == BOX)
      distanceNormal_box(nx,ny,x,y);
    if (_type == SPHERE)
      distanceNormal_sphere(nx,ny,x,y);
  }

  /* Evaluates velocities of the solid at a given point */
  KOKKOS_INLINE_FUNCTION
  double velocityX(double x, double y) const {
    return (_posY - y)*_velTheta + _velX;
  } // velocityX
    
  KOKKOS_INLINE_FUNCTION
  double velocityY(double x, double y) const {
    return (x - _posX)*_velTheta + _velY;
  } // velocityY
    
  KOKKOS_INLINE_FUNCTION
  void velocity(double &vx, double &vy, double x, double y) const {
    vx = velocityX(x, y);
    vy = velocityY(x, y);
  } // velocity
    
  KOKKOS_INLINE_FUNCTION
  void update(double timestep) {
    /* Simple Euler integration - enough for solid bodies, since they
     * are not influenced by the simulation and velocities are typically
     * static
     */
    _posX  += _velX*timestep;
    _posY  += _velY*timestep;
    _theta += _velTheta*timestep;
  } // update
  
}; // class SolidBody

/// typedef to define an array (Kokkos::View) of SolidBody
using SolidBodyList = Kokkos::View<SolidBody*,Device>;

#endif // SOLID_BODIES_H_
