/*
  Copyright (c) 2013 Benedikt Bitterli

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software
  in a product, an acknowledgment in the product documentation would be
  appreciated but is not required.

  2. Altered source versions must be plainly marked as such, and must not be
  misrepresented as being the original software.

  3. This notice may not be removed or altered from any source
  distribution.
*/

#include <algorithm> // for std::swap
#include <math.h>
#include <stdio.h>

#include "../lodepng/lodepng.h"

#include "FluidQuantity.h"
#include "FluidFunctors.h"

//using namespace std;

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
  double *_r; /* Right hand side of pressure solve */
  double *_p; /* Pressure solution */
    
    
  /* Builds the pressure right hand side as the negative divergence */
  void buildRhs() {
    double scale = 1.0/_hx;
        
    for (int y = 0, idx = 0; y < _h; y++) {
      for (int x = 0; x < _w; x++, idx++) {
	_r[idx] = -scale*(_u->at(x + 1, y) - _u->at(x, y) +
			  _v->at(x, y + 1) - _v->at(x, y));
      }
    }
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
    
  /* Applies the computed pressure to the velocity field */
  void applyPressure(double timestep) {
    double scale = timestep/(_density*_hx);
        
    for (int y = 0, idx = 0; y < _h; y++) {
      for (int x = 0; x < _w; x++, idx++) {
	_u->at(x,     y    ) -= scale*_p[idx];
	_u->at(x + 1, y    ) += scale*_p[idx];
	_v->at(x,     y    ) -= scale*_p[idx];
	_v->at(x,     y + 1) += scale*_p[idx];
      }
    }
        
    for (int y = 0; y < _h; y++)
      _u->at(0, y) = _u->at(_w, y) = 0.0;
    for (int x = 0; x < _w; x++)
      _v->at(x, 0) = _v->at(x, _h) = 0.0;
  }
    
public:
  FluidSolver(int w, int h, double density) : _w(w), _h(h), _density(density) {
    _hx = 1.0/std::min(w, h);
        
    _d = new FluidQuantity(_w,     _h,     0.5, 0.5, _hx);
    _u = new FluidQuantity(_w + 1, _h,     0.0, 0.5, _hx);
    _v = new FluidQuantity(_w,     _h + 1, 0.5, 0.0, _hx);
        
    _r = new double[_w*_h];
    _p = new double[_w*_h];
        
    memset(_p, 0, _w*_h*sizeof(double));
  }
    
  ~FluidSolver() {
    delete _d;
    delete _u;
    delete _v;
        
    delete[] _r;
    delete[] _p;
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
};

int main() {
  /* Play with these constants, if you want */
  const int sizeX = 128;
  const int sizeY = 128;
    
  const double density = 0.1;
  const double timestep = 0.005;
    
  unsigned char *image = new unsigned char[sizeX*sizeY*4];

  FluidSolver *solver = new FluidSolver(sizeX, sizeY, density);

  double time = 0.0;
  int iterations = 0;
    
  while (time < 8.0) {
    /* Use four substeps per iteration */
    for (int i = 0; i < 4; i++) {
      solver->addInflow(0.45, 0.2, 0.1, 0.01, 1.0, 0.0, 3.0);
      solver->update(timestep);
      time += timestep;
      fflush(stdout);
    }

    solver->toImage(image);
        
    char path[256];
    sprintf(path, "Frame%05d.png", iterations++);
    lodepng_encode32_file(path, image, sizeX, sizeY);
  }

  return 0;
}
