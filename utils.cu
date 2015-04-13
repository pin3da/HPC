#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdlib>


const int MV = 128;
void fill_random_vec(double *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<double> (rand()) / static_cast<double> (MV);
  }
}

bool cmp_vect(double *a, double *b, int n, double eps = 1e-4) {
  for (int i = 0; i < n; ++i) {
    if (fabs(a[i] - b[i]) > eps)
      return false;
  }
  return true;
}
