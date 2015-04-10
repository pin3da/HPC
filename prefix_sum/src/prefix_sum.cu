#include <cstdio>
#include <cstring>
#include <cassert>

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__-1); exit(-1);}

const int THPB = 2;
const int ELPB = THPB << 1;

__global__ void pref_sum_parallel(int *A, int *B, int n) {
  __shared__ int tmp[ELPB];

  int rid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int idx = (rid << 1);

  if (idx + 1 < n) {
    tmp[tid << 1]  = A[idx];
    tmp[(tid << 1) + 1] = A[idx + 1];
  } else {
    tmp[tid << 1] = tmp[(tid << 1) + 1] = 0;
  }

  int exp = 1;
  for (int d = 1; d < ELPB; d <<= 1, exp <<= 1) {
    __syncthreads();
    int ai = (tid * (d << 1)) + d - 1;
    int bi = (tid * (d << 1)) + (d << 1) - 1;
    if (bi <  ELPB) {
      tmp[bi] += tmp[ai];
    }
  }

  if (tid == 0)
    tmp[ELPB - 1] = 0;

  for (int d = exp >> 1; d > 0; d >>= 1) {
    __syncthreads();
    int ai = (tid * (d << 1)) + d - 1;
    int bi = (tid * (d << 1)) + (d << 1) - 1;
    if (bi < ELPB) {
      int t = tmp[bi];
      tmp[bi] += tmp[ai];
      tmp[ai] = t;
    }
  }

  __syncthreads();
  if (idx + 1 < n) {
    B[idx] = tmp[tid << 1];
    B[idx + 1] = tmp[(tid << 1) + 1];
  }
}

__global__ void group_by_blocks(int *ori, int *A, int *B, int n) {
  int rid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int idx = (rid << 1) + 1;
  int last = (blockIdx.x + 1) * ELPB - 1;

  // Boo... so many unused threads ):
  if (tid == THPB - 1) {
    B[blockIdx.x] = A[idx] + ori[last];
  }

}


__global__ void update_sum(int *A, int *B, int n) {
  int rid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (rid << 1) + 1;
  if (blockIdx.x > 0) {
    B[idx] += A[blockIdx.x];
    B[idx - 1] += A[blockIdx.x];
  }
}

void pref_sum(int *A, int *B, int n) {
  B[0] = 0;
  for (int i = 1; i < n; ++i)
    B[i] = B[i - 1] + A[i - 1];
}

const int MV = 15;
int main() {
  int lengths[] = {16, 1024, 12288, 1048576, 10002432}; // 2048 * X <= 10002432
  srand(time(0));
  for (int tc = 0; tc < 1; ++tc) {
    int n = lengths[tc];
    int *A = new int[n], *B = new int[n];
    for (int i = 0; i < n; ++i)
      A[i] = random() % MV;

    clock_t start = clock();
    pref_sum(A, B, n);
    printf("%d ", n);
    printf("%.10lf ", ((double) (clock() - start) / CLOCKS_PER_SEC) );

    clock_t start_device = clock();
    int *dA, *dB, *dC, *dCT;
    int *ans = new int[n];
    int numBlocks = ((n >> 1) + THPB - 1) / THPB;
    int *pref_b = new int[numBlocks];
    CUDA_CALL(cudaMalloc((void **) &dA, n * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **) &dB, n * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **) &dC, numBlocks * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **) &dCT, numBlocks * sizeof(int)));

    CUDA_CALL(cudaMemcpy(dA, A, n * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dim_block(THPB, 1, 1);
    dim3 dim_grid(numBlocks, 1, 1);
    printf("\nDimgrid : %d \n", numBlocks);

    // computes the prefix sum per block
    pref_sum_parallel <<< dim_grid, dim_block >>> (dA, dB, n);
    group_by_blocks <<< dim_grid, dim_block >>> (dA, dB, dC, n);

    // The number of blocks must be less than the number of elements per block.
    // Otherwise we need to make recursive calls.
    pref_sum_parallel <<< dim3(1, 1, 1), dim_block >>> (dC, dCT, n);
    update_sum <<< dim_grid, dim_block >>> (dCT, dB, n);

    cudaDeviceSynchronize();
    CUDA_CHECK();

    CUDA_CALL( cudaMemcpy(ans, dB, n * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(pref_b, dCT, numBlocks * sizeof(int), cudaMemcpyDeviceToHost) );
    // printf("\n -- %d -- \n", cudaMemcpy(ans, dB, n * sizeof(int), cudaMemcpyDeviceToHost));

#if 1
    for (int i = 0; i < n; ++i) {
      printf("%d ", A[i]);
    }
    puts("");
    for (int i = 0; i < n; ++i) {
      printf("%d ", B[i]);
    }
    puts("");
    for (int i = 0; i < n; ++i) {
      printf("%d ", ans[i]);
    }
    puts("");

    for (int i = 0; i < numBlocks; ++i) {
      printf("%d ", pref_b[i]);
    }
    puts("");
#endif

    bool ok = true;
    int idx;
    for (int i = 0; i < n && ok; ++i)
      if (B[i] != ans[i]) {
        ok = 0;
        idx = i;
      }

    if (!ok) {
      printf("Problem with prefix sum on test case : %d, element %d\n", tc, idx);
      printf("%d != %d\n", B[idx], ans[idx]);
      printf("%d == %d\n", B[idx - 1], ans[idx - 1]);
      exit(1);
    }

    cudaFree(dA);
    cudaFree(dB);
    printf("%.10f\n", (double) (clock() - start_device) / CLOCKS_PER_SEC);

    delete [] A, B, ans, pref_b;
  }

  return 0;
}
