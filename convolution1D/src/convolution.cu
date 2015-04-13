#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <cmath>

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__-1); exit(-1);}

#include "../../utils.cu"


double _filter[]    = {0.006, 0.062, 0.242, 0.383, 0.242, 0.061, 0.006};
int filter_size = 7;

void convolution_seq(double *in, double *filter, double *out, int n, int f_size) {

}

void convolution_par(double *in, double *filter, double *out, int n, int f_size) {

}

void convolution_tiled(double *in, double *filter, double *out, int n, int f_size) {

}

void convolution_const(double *in, double *filter, double *out, int n, int f_size) {

}

int main() {
  srand(time(0));
  int lengths[] = {10, 1024, 1048576};
  int num_test  = 3;
  for (int tc = 0; tc < num_test; ++tc) {
    int n = lengths[tc];
    double *input  = new double[n];
    double *output_hos = new double[n];
    double *output_dev = new double[n];
    fill_random_vec(input, n);
    convolution_seq(input, _filter, output_hos, n, filter_size);
    convolution_par(input, _filter, output_dev, n, filter_size);
    if (!cmp_vect(output_hos, output_dev, n)) {
      fprintf(stderr, "Problem wiht pararallel convolution on test %d\n", tc);
      exit(1);
    }

    convolution_tiled(input, _filter, output_dev, n, filter_size);
    if (!cmp_vect(output_hos, output_dev, n)) {
      fprintf(stderr, "Problem wiht pararallel convolution on test %d\n", tc);
      exit(1);
    }

    convolution_const(input, _filter, output_dev, n, filter_size);
    if (!cmp_vect(output_hos, output_dev, n)) {
      fprintf(stderr, "Problem wiht pararallel convolution on test %d\n", tc);
      exit(1);
    }
  }
  return 0;
}
