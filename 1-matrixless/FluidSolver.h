#ifndef FLUID_SOLVER_H_
#define FLUID_SOLVER_H_

#include "kokkos_shared.h"
#include "FluidQuantity.h"

// =========================================================================
// =========================================================================
/* Fluid solver class. Sets up the fluid quantities, forces incompressibility
 * performs advection and adds inflows.
 */
class FluidSolver {

  /* Fluid quantities */
  FluidQuantity *_d;
  FluidQuantity *_u;
  FluidQuantity *_v;
    
  /* Width and height */
  int _w;
  int _h;
    
  /* Grid cell size and fluid density */
  double _hx;
  double _density;
    
  /* Arrays for: */
  Array2d _r; /* Right hand side of pressure solve */
  Array2d _p; /* Pressure solution */
    
  /* Builds the pressure right hand side as the negative divergence */
  void buildRhs() {

    // call functor
    
  }

  /* Performs the pressure solve using Gauss-Seidel.
   * The solver will run as long as it takes to get the relative error below
   * a threshold, but will never exceed `limit' iterations
   */
  void project(int limit, double timestep) {
    double scale = timestep/(_density*_hx*_hx);
        
    double maxDelta;
    for (int iter = 0; iter < limit; iter++) {
      maxDelta = 0.0;
      for (int y = 0, idx = 0; y < _h; y++) {
	for (int x = 0; x < _w; x++, idx++) {
	  int idx = x + y*_w;
                    
	  double diag = 0.0, offDiag = 0.0;
                    
	  /* Here we build the matrix implicitly as the five-point
	   * stencil. Grid borders are assumed to be solid, i.e.
	   * there is no fluid outside the simulation domain.
	   */
	  if (x > 0) {
	    diag    += scale;
	    offDiag -= scale*_p[idx - 1];
	  }
	  if (y > 0) {
	    diag    += scale;
	    offDiag -= scale*_p[idx - _w];
	  }
	  if (x < _w - 1) {
	    diag    += scale;
	    offDiag -= scale*_p[idx + 1];
	  }
	  if (y < _h - 1) {
	    diag    += scale;
	    offDiag -= scale*_p[idx + _w];
	  }

	  double newP = (_r[idx] - offDiag)/diag;
                    
	  maxDelta = max(maxDelta, fabs(_p[idx] - newP));
                    
	  _p[idx] = newP;
	}
      }

      if (maxDelta < 1e-5) {
	printf("Exiting solver after %d iterations, maximum change is %f\n", iter, maxDelta);
	return;
      }
    }
        
    printf("Exceeded budget of %d iterations, maximum change was %f\n", limit, maxDelta);
  }
    
  void applyPressure(double timestep) {

    // call functor

  }
    
public:
  FluidSolver(int w, int h, double density) :
    _w(w), _h(h),
    _density(density)
  {
    _hx = 1.0/std::min(w, h);
        
    _d = new FluidQuantity(_w,     _h,     0.5, 0.5, _hx);
    _u = new FluidQuantity(_w + 1, _h,     0.0, 0.5, _hx);
    _v = new FluidQuantity(_w,     _h + 1, 0.5, 0.0, _hx);

    // Array2d are ref counted
    _r = Array2d("pressure_rhs",_w,_h);
    _p = Array2d("pressure",_w*_h);
        
  }
    
  ~FluidSolver() {
    delete _d;
    delete _u;
    delete _v;
  }
    
  void update(double timestep) {
    buildRhs();
    project(600, timestep);
    applyPressure(timestep);
        
    _d->advect(timestep, *_u, *_v);
    _u->advect(timestep, *_u, *_v);
    _v->advect(timestep, *_u, *_v);
        
    /* Make effect of advection visible, since it's not an in-place operation */
    _d->flip();
    _u->flip();
    _v->flip();
  }
    
  /* Set density and x/y velocity in given rectangle to d/u/v, respectively */
  void addInflow(double x, double y, double w, double h, double d, double u, double v) {
    _d->addInflow(x, y, x + w, y + h, d);
    _u->addInflow(x, y, x + w, y + h, u);
    _v->addInflow(x, y, x + w, y + h, v);
  }
    
  /* Returns the maximum allowed timestep. Note that the actual timestep
   * taken should usually be much below this to ensure accurate
   * simulation - just never above.
   */
  double maxTimestep() {
    double maxVelocity = 0.0;
    for (int y = 0; y < _h; y++) {
      for (int x = 0; x < _w; x++) {
	/* Average velocity at grid cell center */
	double u = _u->lerp(x + 0.5, y + 0.5);
	double v = _v->lerp(x + 0.5, y + 0.5);
                
	double velocity = sqrt(u*u + v*v);
	maxVelocity = max(maxVelocity, velocity);
      }
    }
        
    /* Fluid should not flow more than two grid cells per iteration */
    double maxTimestep = 2.0*_hx/maxVelocity;
        
    /* Clamp to sensible maximum value in case of very small velocities */
    return std::min(maxTimestep, 1.0);
  }
    
  /* Convert fluid density to RGBA image */
  void toImage(unsigned char *rgba) {
    for (int i = 0; i < _w*_h; i++) {
      int shade = (int)((1.0 - _d->src()[i])*255.0);
      shade = std::max(std::min(shade, 255), 0);
            
      rgba[i*4 + 0] = shade;
      rgba[i*4 + 1] = shade;
      rgba[i*4 + 2] = shade;
      rgba[i*4 + 3] = 0xFF;
    }
  }
}; // class FluidSolver

#endif // FLUID_SOLVER_H_
