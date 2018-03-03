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
#include "FluidSolver.h"

// =========================================================================
// =========================================================================
int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);
  
  //int rank=0;
  //int nRanks=1;
  
  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";
    
  }  
  
  /* Play with these constants, if you want */
  const int sizeX = 128;
  const int sizeY = 128;
    
  const double density = 0.1;
  const double timestep = 0.005;
    
  unsigned char *image = new unsigned char[sizeX*sizeY*4];

  FluidSolver *solver = new FluidSolver(sizeX, sizeY, density);

  double time = 0.0;
  int iterations = 0;

  // init
  solver->init(timestep);

  
  while (time < 4.0) {
    
    /* Use four substeps per iteration */
    for (int i = 0; i < 20; i++) {
      solver->addInflow(0.45, 0.2, 0.15, 0.03, 1.0, 0.0, 3.0);
      solver->update(timestep);
      time += timestep;
      fflush(stdout);
    }

    solver->toImage(image);
        
    char path[256];
    sprintf(path, "Frame%05d.png", iterations++);
    lodepng_encode32_file(path, image, sizeX, sizeY);
  }

  Kokkos::finalize();

  return 0;
}
