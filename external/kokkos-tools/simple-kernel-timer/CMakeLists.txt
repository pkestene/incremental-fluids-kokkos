add_executable(kp_reader
  kp_reader.cpp)

add_library(kp_kernel_timer SHARED
  kp_kernel_timer.cpp
  kp_kernel_info.h)
