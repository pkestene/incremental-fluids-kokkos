set(EXE_NAME 1-Matrixless)

add_executable(${EXE_NAME}
  ${LODEPNG_SRC}
  kokkos_shared.h
  FluidQuantity.h
  FluidFunctors.h
  Fluid.cpp)
target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
target_link_libraries(${EXE_NAME} kokkos dl)
