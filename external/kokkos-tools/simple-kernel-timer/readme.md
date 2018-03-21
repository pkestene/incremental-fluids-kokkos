# A Simple kernel timer

## Purpose

Provides kernel time measurements via kokkos-tools/simple-kernel-timer
The following code is slightly adapted from
https://github.com/kokkos/kokkos-tools

## How to use it

1. run with environment variable KOKKOS_PROFILE_LIBRARY set:
```bash
KOKKOS_PROFILE_LIBRARY=..../external/kokkos-tools/simple-kernel-timer/libkp_kernel_timer.so ./5-curved-boundaries
```

2. to read the results (dump in a .dat file), use the tool kp-reader:
   ```bash
kp_reader ./results.dat
```

