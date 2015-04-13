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

#define THPB 7

double _filter[]    = {0.006, 0.062, 0.242, 0.383, 0.242, 0.061, 0.006};
int filter_size = 7;

void convolution_seq(double *in, double *filter, double *out, int n, int f_size) {
  clock_t start = clock();
  f_size = f_size >> 1;
  for (int i = 0; i < n; ++i) {
    out[i] = 0;
    for (int j = -f_size; j <= f_size; ++j) {
      if (i + j >= 0 && i + j < n) {
        out[i] += in[i + j] * filter[j + f_size];
      }
    }
  }
  printf(" %.10lf ", (double)(clock() - start) / CLOCKS_PER_SEC);
}

__global__ void naive(double *in, double *filter, double *out, int n, int f_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double val = 0;
  int hs = f_size >> 1;
  if (idx < n) {
    for (int i = -hs; i <= hs; ++i) {
      if ((idx + i) >= 0 && (idx + i) < n)
        val += in[idx + i] * filter[i + hs];
    }
    out[idx] = val;
  }
}

__global__ void tiled(double *in, double *filter, double *out, int n, int f_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int hs  =  f_size >> 1;
  int next = (blockIdx.x + 1) * blockDim.x;
  extern __shared__ double tile[];

  if (idx < n) {
    // tile[threadIdx.x + hs] = in[idx];
    out[threadIdx.x + hs] = idx;
  }


  if (idx < hs) {
    out[threadIdx.x] = hs + THPB + threadIdx.x;
  }

  if (threadIdx.x > 3)
    out[hs + THPB + threadIdx.x] = idx;
/*
 *  if (threadIdx.x < hs) {
 *    if (idx >= hs)
 *      // tile[threadIdx.x] = in[blockDim.x * blockIdx.x - hs + threadIdx.x];
 *      out[threadIdx.x] = blockDim.x * blockIdx.x - hs + threadIdx.x;
 *    else
 *      // tile[threadIdx.x] = -1;
 *      out[threadIdx.x] = -1;
 *
 *
 *  }
 *  if (threadIdx.x >= THPB - hs) {
 *    int cur = threadIdx.x - THPB - hs;
 *    if (next + cur < n) {
 *       // tile[cur + THPB + hs] = in[next + cur];
 *       out[cur + THPB + hs] = in[next + cur];
 *    }
 *  }
 *  __syncthreads();
 */

  /*
   *double val = 0;
   *for (int i = 0; i < f_size; ++i) {
   *  if (idx < n)
   *    val += tile[i + threadIdx.x] * filter[i];
   *}
   */

  // if (idx < n)
  //  out[idx] = val;

}

__global__ void const_me(double *in, double *filter, double *out, int n, int f_size) {

}

void convolution_par(double *in, double *filter, double *out, int n, int f_size) {
  clock_t start = clock();
  double *d_in, *d_filter, *d_out;
  CUDA_CALL(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_filter, f_size * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CALL(cudaMemcpy(d_in, in, n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_filter, filter, f_size * sizeof(double), cudaMemcpyHostToDevice));

  dim3 dim_grid((n + THPB - 1) / THPB, 1, 1);
  dim3 dim_block(THPB, 1, 1);

  naive<<< dim_grid, dim_block >>> (d_in, d_filter, d_out, n, f_size);
  cudaDeviceSynchronize();
  CUDA_CHECK();

  CUDA_CALL(cudaMemcpy(out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaFree(d_filter));
  CUDA_CALL(cudaFree(d_out));
  printf(" %.10lf ", (double)(clock() - start) / CLOCKS_PER_SEC);
}

void convolution_tiled(double *in, double *filter, double *out, int n, int f_size) {
  clock_t start = clock();
  double *d_in, *d_filter, *d_out;
  CUDA_CALL(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_filter, f_size * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CALL(cudaMemcpy(d_in, in, n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_filter, filter, f_size * sizeof(double), cudaMemcpyHostToDevice));

  // Kernel cool stuff.
  dim3 dim_grid((n + THPB - 1) / THPB, 1, 1);
  dim3 dim_block(THPB, 1, 1);

  tiled<<< dim_grid, dim_block, (THPB + f_size) * sizeof (double) >>> (d_in, d_filter, d_out, n, f_size);
  cudaDeviceSynchronize();
  CUDA_CHECK();

  CUDA_CALL(cudaMemcpy(out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaFree(d_filter));
  CUDA_CALL(cudaFree(d_out));
  printf(" %.10lf ", (double)(clock() - start) / CLOCKS_PER_SEC);
}

void convolution_const(double *in, double *filter, double *out, int n, int f_size) {
  clock_t start = clock();
  double *d_in, *d_filter, *d_out;
  CUDA_CALL(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_filter, f_size * sizeof(double)));
  CUDA_CALL(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CALL(cudaMemcpy(d_in, in, n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_filter, filter, f_size * sizeof(double), cudaMemcpyHostToDevice));

  // Kernel cool stuff.


  CUDA_CALL(cudaMemcpy(out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_in));
  CUDA_CALL(cudaFree(d_filter));
  CUDA_CALL(cudaFree(d_out));
  printf(" %.10lf ", (double)(clock() - start) / CLOCKS_PER_SEC);
}

void go_out(double *output_hos, double *output_dev, int n) {
#if 1
  puts("");
  for (int i = 0; i < n; ++i)
    printf("%.5lf ", output_hos[i]);
  puts("");
  for (int i = 0; i < n; ++i)
    printf("%.5lf ", output_dev[i]);
  puts("");
#endif
  exit(1);
}

int main() {
  srand(time(0));
  int lengths[] = {13, 1024, 1048576};
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
      fprintf(stderr, "Problem wiht pararallel (naive) convolution on test %d\n", tc);
      go_out(output_hos, output_dev, n);
    }

    convolution_tiled(input, _filter, output_dev, n, filter_size);
    if (!cmp_vect(output_hos, output_dev, n)) {
      fprintf(stderr, "Problem wiht parallel (tiled) convolution on test %d\n", tc);
      go_out(output_hos, output_dev, n);
    }

    // convolution_const(input, _filter, output_dev, n, filter_size);
    if (!cmp_vect(output_hos, output_dev, n)) {
      fprintf(stderr, "Problem wiht pararallel (constant) convolution on test %d\n", tc);
      exit(1);
    }
    delete [] input, output_dev, output_hos;
  }
  return 0;
}
