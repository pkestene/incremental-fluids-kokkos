Incremental fluids with Kokkos
==============================

The present repository is a naive parallelization of the first four steps with the [Kokkos programing model](https://github.com/kokkos/kokkos) for performance portability.

 - step1: setup the basic build blocks of an incompressible flow solver: advection + poisson solver to ensure incompressibility (divergence of velocity is zero)
 - step2: modify advection (third order Runge Kutta)
 - step3: improve the poisson solver: linear solver becomes a preconditioned conjugate gradient: here the preconditionner is not based the modified incomplete Cholesky factorization
          but uses a few Gauss-Seidel SOR iterations instead, easier to parallelize
 - step4: here we slightly deviates from the original step4; we modified the extrapolate operation to be more data parallelism oriented
 - step5: TODO
 - step6: TODO
 - step7: TODO
 - step8: TODO


Illustration of step4:
![step4 with kokkos on GPU](/step4.gif)

## build with Kokkos

Make sure to do a recursive clone of this repository to get also Kokkos source (populated in external subdirectory).

### build with Kokkos / OpenMP backend device

```bash
mkdir build_openmp; cd build_openmp
cmake -DKOKKOS_ENABLE_OPENMP=ON -DKOKKOS_ENABLE_HWLOC=ON ..
```

### build with Kokkos / CUDA backend device

```bash
mkdir build_cuda; cd build_cuda
export CXX=/path/to/nvcc_wrapper
cmake -DKOKKOS_ENABLE_CUDA=ON -DKOKKOS_ENABLE_CUDA_LAMBDA=ON -DKOKKOS_ENABLE_HWLOC=ON -DKOKKOS_ARCH=Maxwell50 ..
```

nvcc_wrapper is located in kokkos sources; here it should be in external/kokkos/bin/nvcc_wrapper

## Performance measurement with Kokkos

TODO

Incremental fluids (original readme)
====================================

Original incremental fluid tutorial is available at https://github.com/tunabrain/incremental-fluids

![Fluid](https://raw.github.com/tunabrain/incremental-fluids/master/Header.png)


The purpose of this project is to provide simple, easy to understand fluid solver implementations in C++, together with code documentation, algorithm explanation and recommended reading. It is meant for people with beginner to intermediate knowledge of computational fluid dynamics looking for working reference implementations to run and study.

This project closely follows Robert Bridson's book, "Fluid Simulation for Computer Graphics", and implements a selection of the methods explained in the book. Ideally, you have a copy of the book sitting on your shelf, which will make it a lot easier to follow along with the code.

The solvers in this project come in a large variety, ranging from minimalistic to complex. All solvers are Eulerian in nature and run on a staggered Marker-and-Cell grid.

The different solvers are sorted into subfolders marked with a number and a short description. Each folder contains a small markdown file explaining the basic ideas behind the code and provides a list of recommended literature to read.

The number of the solver defines a progression - codes with higher number build on codes with lower number, either adding on features or replacing methods with better ones. The basic classes and concepts, however, always stay the same - ideally, you just start with the simplest solver and work your way through to understand and visually confirm the difference in simulation. Code is only explained once when it is introduced and not in any of the successing solvers to avoid clutter.

All solvers are single-file and require no external libraries apart from lodepng to save individual frames. Compilation should be straightforward on all platforms.

If you want, you can also check out a couple of videos rendered with code from this project (just ported to the GPU):

 - http://www.youtube.com/watch?v=D5uqoB13UBM
 - http://www.youtube.com/watch?v=SzqYnjIR4n0
 - http://www.youtube.com/watch?v=e_S-a5VNXbg
 - http://www.youtube.com/watch?v=vMB8elqhum0

