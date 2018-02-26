#ifndef FLUID_SOLVER_H_
#define FLUID_SOLVER_H_

#include "kokkos_shared.h"
#include "FluidQuantity.h"
#include "FluidFunctors.h"

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

  Array2dHost _src_host;
  
  /* Width and height */
  int _w;
  int _h;
    
  /* Grid cell size and fluid density */
  double _hx;
  double _density;
    
  /* Arrays for: */
  Array2d _r;  /* Right hand side of pressure solve */
  Array2d _p;  /* Pressure solution */

  Array2d _z; /* Auxiliary vector */
  Array2d _s; /* Search vector */
  Array2d _precon; /* Preconditioner */
  
  Array2d _aDiag;  /* Matrix diagonal */
  Array2d _aPlusX; /* Matrix off-diagonals */
  Array2d _aPlusY;
  
  /* Builds the pressure right hand side as the negative divergence */
  void buildRhs() {

    double scale = 1.0/_hx;

    BuildRHSFunctor::apply(_r, _u->_src, _v->_src, scale, _w, _h);
    
  } // buildRhs

  /* Builds the pressure matrix. Since the matrix is very sparse and
   * symmetric, it allows for memory friendly storage.
   */
  void buildPressureMatrix(double timestep) {
    double scale = timestep/(_density*_hx*_hx);

    // reset Array
    reset_view(_aDiag);
    
    // buildPressureMatrix
    // serail version refactored to avoid data-race
    BuildPressureMatrixFunctor::apply(_aDiag, _aPlusX, _aPlusY, scale, _w, _h);

  } // buildPressureMatrix

  /* Builds the modified incomplete Cholesky preconditioner */
  void buildPreconditioner() {
    const double tau = 0.97;
    const double sigma = 0.25;

    BuildPreconditionerFunctor::apply(_aDiag, _aPlusX, _aPlusY, _precon, _w, _h, tau, sigma);
    
  } // buildPreconditionner

  
  /* Performs the pressure solve using a conjugate gradient method.
   * The solver will run as long as it takes to get the relative error below
   * a threshold, but will never exceed `limit' iterations
   */
   void project(int limit, double timestep) {

    double scale = timestep/(_density*_hx*_hx);
        
    double maxDelta;
    for (int iter = 0; iter < limit; iter++) {
      maxDelta = 0.0;

      // if (iteration_type == ITER_GAUSS_SEIDEL) {
      // 	ProjectFunctor_GaussSeidel::apply(_p, _r, scale, maxDelta, _w, _h);
      // } else {
      // 	ProjectFunctor_Jacobi::apply(_p, _p2, _r, scale, maxDelta, _w, _h);
      // 	// the following deep copy could be avoided by swapping _p and _p2
      // 	Kokkos::deep_copy(_p,_p2);
      // }
      
      if (maxDelta < 1e-5) {
    	printf("Exiting solver after %d iterations, maximum change is %f\n", iter, maxDelta);
    	return;
      }
    } // for iter

    
    printf("Exceeded budget of %d iterations, maximum change was %f\n", limit, maxDelta);
              
  } // project
    
  void applyPressure(double timestep) {

    double scale = timestep/(_density*_hx);

    ApplyPressureFunctor::apply(_p, _u->_src, _v->_src, scale, _w, _h);

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

    _src_host = Array2dHost("data_on_host",_w,_h);
    
    // Array2d are ref counted
    _r  = Array2d("pressure_rhs",_w,_h);
    _p  = Array2d("pressure"    ,_w,_h);

    _z = Array2d("aux_vector",_w,_h);
    _s = Array2d("search_vector",_w,_h);
    _precon = Array2d("preconditioner",_w,_h);
    
    _aDiag = Array2d("matrix_diagonal",_w,_h);
    _aPlusX = Array2d("off_diag_x",_w,_h);
    _aPlusY = Array2d("off_diag_y",_w,_h);
    

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
    
    AdvectionFunctor::apply(*_d,*_u,*_v,timestep);
    AdvectionFunctor::apply(*_u,*_u,*_v,timestep);
    AdvectionFunctor::apply(*_v,*_u,*_v,timestep);
        
    /* Make effect of advection visible, since it's not an in-place operation */
    _d->flip();
    _u->flip();
    _v->flip();
  }
    
  /* Set density and x/y velocity in given rectangle to d/u/v, respectively */
  void addInflow(double x, double y, double w, double h, double d, double u, double v) {

    InflowFunctor::apply(*_d, d, x, y, x + w, y + h);
    InflowFunctor::apply(*_u, u, x, y, x + w, y + h);
    InflowFunctor::apply(*_v, v, x, y, x + w, y + h);

  }
    
  /* Returns the maximum allowed timestep. Note that the actual timestep
   * taken should usually be much below this to ensure accurate
   * simulation - just never above.
   *
   * Not used.
   */
  // double maxTimestep() {
    
  //   double maxVelocity = 0.0;

  //   MaxVelocityFunctor::apply(*_u, *_v, maxVelocity, _w, _h);
    
  //   /* Fluid should not flow more than two grid cells per iteration */
  //   double maxTimestep = 2.0*_hx/maxVelocity;
        
  //   /* Clamp to sensible maximum value in case of very small velocities */
  //   return std::min(maxTimestep, 1.0);
    
  // }
    
  /* Convert fluid density to RGBA image */
  void toImage(unsigned char *rgba) {

    // copy current density array to a temporary destination on "host"
    // which can then be used to save data to a file
    Kokkos::deep_copy(_src_host, _d->_src);

    double *data = _src_host.ptr_on_device();
    
    for (int i = 0; i < _w*_h; i++) {
      int shade = (int)((1.0 - data[i])*255.0);
      shade = std::max(std::min(shade, 255), 0);
            
      rgba[i*4 + 0] = shade;
      rgba[i*4 + 1] = shade;
      rgba[i*4 + 2] = shade;
      rgba[i*4 + 3] = 0xFF;
    }
  }
}; // class FluidSolver

#endif // FLUID_SOLVER_H_
