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

  int rid = blockIdx.x * THPB + threadIdx.x;
  int tid = threadIdx.x;
  int idx = (rid << 1);

  if (idx + 1 < n) {
    tmp[tid << 1]  = A[idx];
    tmp[(tid << 1) + 1] = A[idx + 1];
  } else {
    tmp[tid << 1] = tmp[(tid << 1) + 1] = 0;
  }

  for (int d = 1; d <= ELPB; d <<= 1) {
    __syncthreads();
    int ai = (tid << 1) + d - 1;
    int bi = (tid << 1) + (d << 1) - 1;
    if (bi <  ELPB) {
      tmp[bi] += tmp[ai];
    }
  }

  if (idx + 1< n) {
    B[idx] = tmp[tid << 1];
    B[idx + 1] = tmp[(tid << 1) + 1];
  }
}

void pref_sum(int *A, int *B, int n) {
  B[0] = A[0];
  for (int i = 1; i < n; ++i)
    B[i] = B[i - 1] + A[i];
}

const int MV = 15;
int main() {
  int lengths[] = {8, 10000, 100000, 10000000};
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
    int *dA, *dB;
    int *ans = new int[n];
    CUDA_CALL(cudaMalloc((void **) &dA, n * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **) &dB, n * sizeof(int)));

    CUDA_CALL(cudaMemcpy(dA, A, n * sizeof(int), cudaMemcpyHostToDevice));

    int block_size = THPB;
    dim3 dim_block(block_size, 1, 1);
    dim3 dim_grid(((n >> 1) + block_size - 1) / block_size, 1, 1);
    printf("\nDimgrid : %d \n", ((n >> 1) + block_size - 1) / block_size);
    pref_sum_parallel<<<dim_grid, dim_block>>>(dA, dB, n);
    cudaDeviceSynchronize();
    CUDA_CHECK();

    printf("\n%d\n", n);

    CUDA_CALL( cudaMemcpy(ans, dB, n * sizeof(int), cudaMemcpyDeviceToHost) );
    // printf("\n -- %d -- \n", cudaMemcpy(ans, dB, n * sizeof(int), cudaMemcpyDeviceToHost));


    for (int i = 0; i < n; ++i) {
      printf("%d ", A[i]);
    }
    puts("");
    for (int i = 0; i < n; ++i) {
      printf("%d ", ans[i]);
    }
    puts("");
/*
 *    bool ok = true;
 *    for (int i = 0; i < n && ok; ++i)
 *      if (B[i] != ans[i])
 *        ok = 0;
 *
 *
 *    if (!ok) {
 *      printf("Problem with prefix sum on test case : %d\n", n);
 *      exit(1);
 *    }
 *
 *    cudaFree(dA);
 *    cudaFree(dB);
 */
    printf("%.10f\n", (double) (clock() - start_device) / CLOCKS_PER_SEC);

    delete [] A, B, ans;
  }

  return 0;
}
