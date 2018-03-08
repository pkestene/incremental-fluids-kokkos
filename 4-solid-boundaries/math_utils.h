#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

// some utilities

KOKKOS_INLINE_FUNCTION
int imin(int i, int j) {
  return i<j ? i : j;
} // imin

KOKKOS_INLINE_FUNCTION
int imax(int i, int j) {
  return i<j ? j : i;
} // imax

/* See http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c */
template <typename T>
KOKKOS_INLINE_FUNCTION
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/* Non-zero sgn */
template <typename T>
KOKKOS_INLINE_FUNCTION
int nsgn(T val) {
    return (val < T(0) ? -1 : 1);
}

/* Length of vector (x, y) */
KOKKOS_INLINE_FUNCTION
double length(double x, double y) {
  return sqrt(x*x + y*y);
}


#endif // MATH_UTILS_
