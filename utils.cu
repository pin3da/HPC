#include <ctime>
#include <cmath>

void fill_random_vec(double *a, int n) {
}

bool cmp_vect(double *a, double *b, int n, double eps = 1e-4) {
  for (int i = 0; i < n; ++i) {
    if (fabs(a[i] - b[i]) > eps)
      return false;
  }
  return true;
}
