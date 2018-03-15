cmake_minimum_required (VERSION 3.1)

project(incremental-fluids-kokkos CXX C)

# C++11 is for Kokkos
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# external sources : kokkos
#
add_subdirectory(external/kokkos)

# pass Kokkos include directories to our target application
include_directories(${Kokkos_INCLUDE_DIRS_RET})

set(LODEPNG_SRC "../lodepng/lodepng.h" "../lodepng/lodepng.cpp")

add_subdirectory(1-matrixless)
add_subdirectory(2-better-advection)
add_subdirectory(3-conjugate-gradients)
add_subdirectory(3-conjugate-gradients_GS)
add_subdirectory(4-solid-boundaries)
add_subdirectory(5-curved-boundaries)
# add_subdirectory(6-heat)
# add_subdirectory(7-variable-density)
# add_subdirectory(8-flip)
