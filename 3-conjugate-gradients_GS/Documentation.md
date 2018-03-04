3) Conjugate Gradient Method - revisited
========================================

This is exactly the same as in step3 but we change the preconditionning from "Modified Incomplete Cholesky" (MIC) to "Red-Black Gauss-Seidel" (a few iteration of it).
MIC is really hard to parallelize for multi-thread and massively multi-thread architectures.

Ref:
- A list of GPU-friendly preconditioners can be found in "GPU-accelerated Preconditioned Iterative Linear Solvers", by R. Li and Y. Saad:
http://www-users.cs.umn.edu/~saad/PDF/umsi-2010-112.pdf
- going further, see a Task Dag parallel version of MIC precontioning:
https://trilinos.org/wordpress/wp-content/uploads/2016/01/KokkosPortableAPI.pdf