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
  clock_t start = clock();
  f_size = f_size >> 1;
  for (int i = 0; i < n; ++i) {
    out[i] = 0;
    for (int j = -f_size; j <= f_size; ++j) {
      if (i + j > 0 && i + j < n) {
        out[i] += in[i + j] * filter[j];
      }
    }
  }
  printf(" %.10lf ", (double)(clock() - start) / CLOCKS_PER_SEC);
}

void convolution_par(double *in, double *filter, double *out, int n, int f_size) {
  double *d_in, *d_filter, *d_out;
  CUDA_CALL(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_filter, f_size * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CALL(cudaMemcpy(d_in, in, n * sizeof(double)));
  CUDA_CALL(cudaMemcpy(d_filter, filter, f_size * sizeof(double)));

  // Kernel cool stuff.


  CUDA_CALL(cudaMemcpy(out, d_out, n * sizeof(double)));

  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaFree(d_filter));
  CUDA_CALL(cudaFree(d_out));
}

void convolution_tiled(double *in, double *filter, double *out, int n, int f_size) {
  double *d_in, *d_filter, *d_out;
  CUDA_CALL(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_filter, f_size * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CALL(cudaMemcpy(d_in, in, n * sizeof(double)));
  CUDA_CALL(cudaMemcpy(d_filter, filter, f_size * sizeof(double)));

  // Kernel cool stuff.


  CUDA_CALL(cudaMemcpy(out, d_out, n * sizeof(double)));

  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaFree(d_filter));
  CUDA_CALL(cudaFree(d_out));
}

void convolution_const(double *in, double *filter, double *out, int n, int f_size) {
  double *d_in, *d_filter, *d_out;
  CUDA_CALL(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_filter, f_size * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CALL(cudaMemcpy(d_in, in, n * sizeof(double)));
  CUDA_CALL(cudaMemcpy(d_filter, filter, f_size * sizeof(double)));

  // Kernel cool stuff.


  CUDA_CALL(cudaMemcpy(out, d_out, n * sizeof(double)));

  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaFree(d_filter));
  CUDA_CALL(cudaFree(d_out));
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
