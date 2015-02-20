#include <cstdio>
#include <cstring>
#include <cassert>

__global__ void multiply_parallel(int *A, int *B, int *C, int n, int m, int o) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && j < o) {
    C[i * o + j] = 0;
    for (int k = 0; k < m; ++k) {
     C[i * o + j] += A[i * m + k] * B[k * o + j];
    }
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
    printf("%d %d %d\n", n, m, o);
    int *A = new int[n * m], *B = new int[m * o], *C = new int[n * o];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        scanf("%d", A + (i * m + j));

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < o; ++j)
        scanf("%d", B + (i * o + j));;

    clock_t start = clock();
    multiply(A, B, C, n, m, o);
    printf("Serial multiplication : %.10lf\n", ((double) (clock() - start) / CLOCKS_PER_SEC) );


    cudaEvent_t start_device, stop_device;
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
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

    int block_size = 32;
    dim3 dim_block(block_size, block_size, 1);
    dim3 dim_grid((n + block_size - 1) / block_size, (o + block_size - 1) / block_size, 1);

    multiply_parallel<<<dim_grid, dim_block>>>(dA, dB, dC, n, m, o);
    cudaDeviceSynchronize();
    // cudaEventRecord(stop_device, 0);
    // cudaEventSynchronize(stop_device);

    if (cudaSuccess != cudaMemcpy(ans, dC, n * o * sizeof(int), cudaMemcpyDeviceToHost)) {
      puts("Problem copying memory to host");
      exit(1);
    }

#ifdef DEBUG
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
#endif

    bool ok = true;
    for (int i = 0; i < (n * o) && ok; ++i) {
      if (C[i] != ans[i])
        ok = 0;
    }

    if (ok)
      puts("Correct (:");
    else
      puts("Next try ):");


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_device, stop_device);
    cudaEventDestroy(start_device);
    cudaEventDestroy(stop_device);
    printf("Parallel multiplication : %.10f\n", elapsed_time);

    delete [] A, B , C, ans;

  }
  return 0;
}
