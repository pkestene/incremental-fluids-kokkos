set(EXE_NAME 4-solid-boundaries)

add_executable(${EXE_NAME}
  ${LODEPNG_SRC}
  kokkos_shared.h
  FluidQuantity.h
  SolidBody.h
  FluidFunctors.h
  Fluid.cpp)
target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
target_link_libraries(${EXE_NAME} kokkos dl)
