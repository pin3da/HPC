#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda.h>

#define TILE_WIDTH 8
const double eps = 1e-4;

__global__ void multiply_parallel(float *A, float *B, float *C, int n, int m, int o) {

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float pvalue = 0;
  int top = (m + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int tm = 0; tm < top; ++tm) {
    if (row < n && (tm * TILE_WIDTH + tx) < m)
      Mds[ty][tx] = A[row * m  + tm * TILE_WIDTH + tx];
    else
      Mds[ty][tx] = 0.0;

    if ((tm * TILE_WIDTH + ty) < m && col < o)
      Nds[ty][tx] = B[(tm * TILE_WIDTH + ty) * o + col];
    else
      Nds[ty][tx] = 0.0;

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      // for (int k = 0; k < m; ++k) {
        // pvalue += A[i * m + k] * B[k * o + j];
      // if (row < n && (tm * TILE_WIDTH + tx) < m && (tm * TILE_WIDTH + ty) < m && col < o)
        pvalue += Mds[ty][k] * Nds[k][tx];
      // }
    }
    __syncthreads();

    if (row < n && col < o)
      C[row * o  + col] = pvalue;

  }
}

void multiply(float *A, float *B, float *C, int n, int m, int o) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < o; ++j) {
      C[i * o + j] = 0.0;
      for (int k = 0; k < m; ++k) {
        C[i * o + j] += A[i * m + k] * B[k * o + j];
      }
    }
  }
}

int main() {
  int tc;
  scanf("%d", &tc);
  while (tc--) {
    int n, m, o;
    scanf("%d%d%d", &n, &m, &o);
    float *A = new float[n * m], *B = new float[m * o], *C = new float[n * o];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        scanf("%f", A + (i * m + j));

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < o; ++j)
        scanf("%f", B + (i * o + j));

    clock_t start = clock();
    multiply(A, B, C, n, m, o);
    printf("%lld ", (long long)n * (long long)m * (long long)o);
    printf("%.10lf ", ((double) (clock() - start) / CLOCKS_PER_SEC) );


    clock_t start_device = clock();
    float *dA, *dB, *dC;
    float *ans = new float[n * o];
    if (cudaSuccess != cudaMalloc((void **) &dA, n * m * sizeof(float))) {
      puts("Problem allocating memory in device");
      exit(1);
    }

    if (cudaSuccess != cudaMalloc((void **) &dB, m * o * sizeof(float))) {
      puts("Problem allocating memory in device");
      exit(1);
    }

    if (cudaSuccess != cudaMalloc((void **) &dC, n * o * sizeof(float))) {
      puts("Problem allocating memory in device");
      exit(1);
    }

    if (cudaSuccess != cudaMemcpy(dA, A, n * m * sizeof(float), cudaMemcpyHostToDevice)) {
      puts("Problem copying memory to device");
      exit(1);
    }

    if (cudaSuccess != cudaMemcpy(dB, B, m * o * sizeof(float), cudaMemcpyHostToDevice)) {
      puts("Problem copying memory to device");
      exit(1);
    }

    int block_size = TILE_WIDTH;
    dim3 dim_block(block_size, block_size, 1);
    dim3 dim_grid((o + block_size - 1) / block_size, (n + block_size - 1) / block_size, 1);

    //matrixMulKernelTiled<<<dim_grid, dim_block>>>(dA, dB, dC, n, m, o);
    multiply_parallel<<<dim_grid, dim_block>>>(dA, dB, dC, n, m, o);
    cudaDeviceSynchronize();

    if (cudaSuccess != cudaMemcpy(ans, dC, n * o * sizeof(float), cudaMemcpyDeviceToHost)) {
      puts("Problem copying memory to host");
      exit(1);
    }

    bool ok = true;
    for (int i = 0; i < (n * o) && ok; ++i) {
      if (fabs(C[i] -  ans[i]) > eps)
        ok = 0;
    }

    if (!ok) {
      printf("Problem with matrix %d %d %d\n", n, m, o);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < o; ++j) {
          printf("%.10f ", C[i * o + j]);
        }
        puts("");
      }

      puts("");

      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < o; ++j) {
          printf("%.10f ", ans[i * o + j]);
        }
        puts("");
      }
      exit(1);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    printf("%.10lf\n", (double) (clock() - start_device) / CLOCKS_PER_SEC);

    delete [] A, B , C, ans;

  }
  return 0;
}
