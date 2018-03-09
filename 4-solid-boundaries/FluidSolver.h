#ifndef FLUID_SOLVER_H_
#define FLUID_SOLVER_H_

#include "kokkos_shared.h"
#include "FluidQuantity.h"
#include "FluidFunctors.h"
#include "SolidBody.h"


/* Computes `dst' = `a' + `b'*`s' */
void scaledAdd(Array2d dst, Array2d a, Array2d b, double s) {
  
  const int size = dst.dimension_0()*dst.dimension_1();
  Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int index) {
      int x, y;
      index2coord(index,x,y,dst.dimension_0(),dst.dimension_1());
      dst(x,y) = a(x,y) + b(x,y)*s;
    });
  
} // scaledAdd

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
  Array2dHost_uchar _cell_host;
  
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

  SolidBodyList _bodies;
  
  /* Builds the pressure right hand side as the negative divergence */
  void buildRhs() {

    double scale = 1.0/_hx;

    BuildRHSFunctor::apply(_r, _u->_src, _v->_src, _d->_cell, scale, _w, _h);
    
  } // buildRhs

  /* Builds the pressure matrix. Since the matrix is very sparse and
   * symmetric, it allows for memory friendly storage.
   *
   * Called only once at the begin of simulation.
   */
  void buildPressureMatrix(double timestep) {
    double scale = timestep/(_density*_hx*_hx);

    // reset Array
    reset_view(_aDiag);
    
    // buildPressureMatrix
    // serial version refactored to avoid data-race
    BuildPressureMatrixFunctor::apply(_aDiag, _aPlusX, _aPlusY, _d->_cell, scale, _w, _h);    
  } // buildPressureMatrix


  /* Apply preconditioner to vector `a' and store it in `dst' */
  void applyPreconditioner(Array2d dst, Array2d a) {

    double omega = 1.5;
    int nbIter = 10;
    reset_view(dst);
    ApplyPreconditionerFunctor::apply(dst, a, _d->_cell, _w, _h, omega, nbIter);
    
  } // applyPreconditioner

  /* Returns the dot product of vectors `a' and `b' */
  double dotProduct(Array2d a, Array2d b) {
    double result = 0.0;

    DotProductFunctor::apply(a,b,result);
    
    return result;
    
  } // dotProduct

  /* Multiplies internal pressure matrix with vector `b' 
   * and stores the result in `dst' */
  void matrixVectorProduct(Array2d dst, Array2d b) {

    MatrixVectorProductFunctor::apply(dst,b,_aDiag,_aPlusX,_aPlusY);
    
  }


  /* Returns maximum absolute value in vector `a' */
  double infinityNorm(Array2d a) {

    double maxA = 0.0;

    InfinityNormFunctor::apply(a,maxA);

    return maxA;
    
  } // inifinityNorm
  
  /* Performs the pressure solve using a conjugate gradient method.
   * The solver will run as long as it takes to get the relative error below
   * a threshold, but will never exceed `limit' iterations
   */
   void project(int limit, double timestep) {

     reset_view(_p); /* Initial guess of zeroes */
     applyPreconditioner(_z, _r);
     Kokkos::deep_copy(_s,_z);
     
     double maxError = infinityNorm(_r);
     if (maxError < 1e-5)
       return;
     
     double sigma = dotProduct(_z, _r);
     for (int iter = 0; iter < limit; iter++) {
       matrixVectorProduct(_z, _s);

       double alpha = sigma/dotProduct(_z, _s);
       scaledAdd(_p, _p, _s, alpha);
       scaledAdd(_r, _r, _z, -alpha);
       
       maxError = infinityNorm(_r);
       if (maxError < 1e-5) {
	 printf("Exiting solver after %d iterations, maximum error is %.15f\n", iter, maxError);
	 return;
       }

       applyPreconditioner(_z, _r);

       double sigmaNew = dotProduct(_z, _r);
       scaledAdd(_s, _z, _s, sigmaNew/sigma);
       sigma = sigmaNew;
      
    } // for iter

     printf("Exceeded budget of %d iterations, maximum error was %.15f\n", limit, maxError);
              
  } // project
    
  void applyPressure(double timestep) {

    double scale = timestep/(_density*_hx);

    ApplyPressureFunctor::apply(_p, _u->_src, _v->_src, _d->_cell, scale, _w, _h);

  }

  void setBoundaryCondition() {

    SetBoundaryConditionFunctor::apply(_u->_src,
				       _v->_src,
				       _d->_cell,
				       _d->_body,
				       _bodies,
				       _w, _h, _hx);
    
  }
    
public:
  FluidSolver(int w, int h, double density, SolidBodyList bodies) :
    _w(w), _h(h),
    _density(density),
    _bodies(bodies)
  {
    _hx = 1.0/std::min(w, h);
        
    _d = new FluidQuantity(_w,     _h,     0.5, 0.5, _hx);
    _u = new FluidQuantity(_w + 1, _h,     0.0, 0.5, _hx);
    _v = new FluidQuantity(_w,     _h + 1, 0.5, 0.0, _hx);

    _src_host = Array2dHost("data_on_host",_w,_h);
    _cell_host = Array2dHost_uchar("cell_on_host",_w,_h);
    
    // Array2d are ref counted
    _r  = Array2d("pressure_rhs",_w,_h);
    _p  = Array2d("pressure"    ,_w,_h);

    _z = Array2d("aux_vector",_w,_h);
    _s = Array2d("search_vector",_w,_h);

    _aDiag = Array2d("matrix_diagonal",_w,_h);
    _aPlusX = Array2d("off_diag_x",_w,_h);
    _aPlusY = Array2d("off_diag_y",_w,_h);
    
  }
    
  ~FluidSolver() {
    delete _d;
    delete _u;
    delete _v;
  }

  /**
   * called once before entering the time loop
   */
  void init(double timestep) {

    // pressure matrix never changed, so build it here
    // computes aDiag, aPlusX, aPlusY
    buildPressureMatrix(timestep);    

  }
  
  void update(double timestep) {

    // fill solidFields - TODO
    //FillSolidFields(_bodies);
    //FillSolidFields(_bodies);
    //FillSolidFields(_bodies);
    
    setBoundaryCondition();
    
    // compute scale divergence of velocity field
    buildRhs();

    // compute pressure
    project(600, timestep);

    // update velocity field with pressure gradient
    applyPressure(timestep);

    // extrapolate functor TODO
    //ExtrapolateFunctor::apply();
    //ExtrapolateFunctor::apply();
    //ExtrapolateFunctor::apply();
    
    // update velocity field with advection
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
    Kokkos::deep_copy(_cell_host, _d->_cell);

    double *data = _src_host.ptr_on_device();
    uint8_t *cell = _cell_host.ptr_on_device();
    
    for (int i = 0; i < _w*_h; i++) {
      int shade = (int)((1.0 - data[i])*255.0);
      shade = std::max(std::min(shade, 255), 0);

      if (cell[i] == CELL_SOLID)
	shade = 0.0;
      
      rgba[i*4 + 0] = shade;
      rgba[i*4 + 1] = shade;
      rgba[i*4 + 2] = shade;
      rgba[i*4 + 3] = 0xFF;
    }
  }
}; // class FluidSolver

#endif // FLUID_SOLVER_H_
