#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda.h>

#define TILE_WIDTH 4

__global__ void matrixMulKernelTiled(int *d_M, int *d_N, int *d_P, int rowM, int colM , int colN){
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int Pvalue = 0;

    for(int m = 0; m < ceil(colM / float(TILE_WIDTH)); ++m){
        if(m * TILE_WIDTH + tx < colM && row < rowM){
          Mds[ty][tx] = d_M[row*colM + m*TILE_WIDTH + tx];
        }else{
            Mds[ty][tx] = 0.0;
        }
        if(m*TILE_WIDTH + ty < colM && col < colN){
            Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty) * colN + col];
        }else{
            Nds[ty][tx] =0.0;
        }
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (row < rowM && col < colN)
        d_P[row*colN+col] = Pvalue;
}



__global__ void multiply_parallel(int *A, int *B, int *C, int n, int m, int o) {

  __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  int pvalue = 0;
  int top = (m + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int tm = 0; tm < top; ++tm) {
    if (row < n && (tm * TILE_WIDTH + tx) < m)
      Mds[ty][tx] = A[row * m  + tm * TILE_WIDTH + tx];
    else
      Mds[ty][tx] = 0;

    if ((tm * TILE_WIDTH + ty) < m && col < o)
      Nds[ty][tx] = B[(tm * TILE_WIDTH + ty) * o + col];
    else
      Nds[ty][tx] = 0;

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

void multiply(int *A, int *B, int *C, int n, int m, int o) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < o; ++j) {
      C[i * o + j] = 0;
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
    int *A = new int[n * m], *B = new int[m * o], *C = new int[n * o];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        scanf("%d", A + (i * m + j));

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < o; ++j)
        scanf("%d", B + (i * o + j));

    clock_t start = clock();
    multiply(A, B, C, n, m, o);
    printf("%lld ", (long long)n * (long long)m * (long long)o);
    printf("%.10lf ", ((double) (clock() - start) / CLOCKS_PER_SEC) );


    clock_t start_device = clock();
    int *dA, *dB, *dC;
    int *ans = new int[n * o];
    if (cudaSuccess != cudaMalloc((void **) &dA, n * m * sizeof(int))) {
      puts("Problem allocating memory in device");
      exit(1);
    }

    if (cudaSuccess != cudaMalloc((void **) &dB, m * o * sizeof(int))) {
      puts("Problem allocating memory in device");
      exit(1);
    }

    if (cudaSuccess != cudaMalloc((void **) &dC, n * o * sizeof(int))) {
      puts("Problem allocating memory in device");
      exit(1);
    }

    if (cudaSuccess != cudaMemcpy(dA, A, n * m * sizeof(int), cudaMemcpyHostToDevice)) {
      puts("Problem copying memory to device");
      exit(1);
    }

    if (cudaSuccess != cudaMemcpy(dB, B, m * o * sizeof(int), cudaMemcpyHostToDevice)) {
      puts("Problem copying memory to device");
      exit(1);
    }

    int block_size = TILE_WIDTH;
    dim3 dim_block(block_size, block_size, 1);
    dim3 dim_grid((o + block_size - 1) / block_size, (n + block_size - 1) / block_size, 1);

    //matrixMulKernelTiled<<<dim_grid, dim_block>>>(dA, dB, dC, n, m, o);
    multiply_parallel<<<dim_grid, dim_block>>>(dA, dB, dC, n, m, o);
    cudaDeviceSynchronize();

    if (cudaSuccess != cudaMemcpy(ans, dC, n * o * sizeof(int), cudaMemcpyDeviceToHost)) {
      puts("Problem copying memory to host");
      exit(1);
    }

    bool ok = true;
    for (int i = 0; i < (n * o) && ok; ++i) {
      if (C[i] != ans[i])
        ok = 0;
    }

    if (!ok) {
      printf("Problem with matrix %d %d %d\n", n, m, o);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < o; ++j) {
          printf("%d ", C[i * o + j]);
        }
        puts("");
      }

      puts("");

      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < o; ++j) {
          printf("%d ", ans[i * o + j]);
        }
        puts("");
      }
      exit(1);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    printf("%.10f\n", (double) (clock() - start_device) / CLOCKS_PER_SEC);

    delete [] A, B , C, ans;

  }
  return 0;
}
