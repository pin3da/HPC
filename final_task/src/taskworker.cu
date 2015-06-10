#include <time.h>
#include <stdlib.h>
#include <czmq.h>
#include <iostream>
#include <map>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;
const int THPB = 1024;

typedef long long int LL;
typedef pair<LL, LL> PLL;

__constant__ LL g_powers[64];

PLL ROU[] = {make_pair(1711276033LL, 1223522572LL), make_pair(1790967809LL, 1110378081LL)};

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void ext_euclid(LL a, LL b, LL &x, LL &y, LL &g) {
  x = 0, y = 1, g = b;
  LL m, n, q, r;
  for (LL u = 1, v = 0; a != 0; g = a, a = r) {
    q = g / a, r = g % a;
    m = x - u * q, n = y - v * q;
    x = u, y = v, u = m, v = n;
  }
}

LL mod_inv(LL n, LL m) {
  LL x, y, gcd;
  ext_euclid(n, m, x, y, gcd);
  if (gcd != 1)
    return 0;
  return (x + m) % m;
}

int bit_reverse(int x, int n) {
  int ans = 0;
  for (int i = 0; i < n; i++)
    if ((x >> i) & 1)
      ans |= ((1 << (n - i - 1)));
  return ans;
}

void bit_reverse_copy(LL *a, LL *A, int n, int length) {
  for (int i = 0; i < length; i++)
    A[bit_reverse(i, n)] = a[i];
}

void compute_powers(LL *powers, int ln, LL basew, LL prime){
  powers[0] = basew;
  for (int i = 1; i < ln; i++){
    powers[i] = (powers[i - 1] * powers[i - 1]) % prime;
  }
}

void fft(LL *a, LL *A, int dir, LL prime, LL basew, int length) {
  int ln = ceil(log2(float(length)));
  bit_reverse_copy(a, A, ln, length);
  LL *powers = (LL*) malloc (sizeof (LL) * ln);
  compute_powers(powers, ln, basew, prime);

  for (int s = 1; s <= ln; s++) {
    long long m = (1LL << s);
    LL wm = powers[ln - s];
    if (dir < 0)
      wm = mod_inv(wm, prime);

    for (int k = 0; k < length; k += m) {
      LL w = 1, mh = m >> 1;
      for (int j = 0; j < mh; j++) {
        LL t = (w * A[k + j + mh]) % prime;
        LL u = A[k + j];
        A[k + j] = (u + t) % prime;
        A[k + j + mh] = (u - t + prime) % prime;
        w = (w * wm) % prime;
      }
    }
  }

  if(dir < 0){
    for (int i = 0; i < length; ++i) {
      A[i] = (A[i] * mod_inv(length, prime)) % prime;
    }
  }
}

__device__ void d_ext_euclid(LL a, LL b, LL &x, LL &y, LL &g) {
  x = 0, y = 1, g = b;
  LL m, n, q, r;
  for (LL u = 1, v = 0; a != 0; g = a, a = r) {
    q = g / a, r = g % a;
    m = x - u * q, n = y - v * q;
    x = u, y = v, u = m, v = n;
  }
}

__device__ LL d_mod_inv(LL n, LL m) {
  LL x, y, gcd;
  d_ext_euclid(n, m, x, y, gcd);
  if (gcd != 1)
    return 0;
  return (x + m) % m;
}


__global__ void fft_kernel (LL *A, int dir, LL prime, int ln, int length, int s) {
  int pos = threadIdx.x + blockDim.x * blockIdx.x;
  LL m = (1LL << s);
  LL wm = g_powers[ln - s];
  LL k = pos * m;
  if (dir < 0)
    wm = d_mod_inv (wm, prime);

  if (k >= length)
    return;

  LL w = 1;
  LL mh = m >> 1;
  for (int j = 0; j < mh; j++){
    LL t = (w * A[k + j + mh]) % prime;
    LL u = A[k + j];
    A[k + j] = (u + t) % prime;
    A[k + j + mh] = (u - t + prime) % prime;
    w = (w * wm) % prime;
  }
}

__global__ void convolution_parallel(LL *d_A, LL *d_B, LL prime, int length) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= length)
    return;
  d_A[pos] = (d_A[pos] * d_B[pos]) % prime;;


}

__global__ void divide_parallel(LL *d_A, LL prime, int length) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= length)
    return;
  LL inv = d_mod_inv(length, prime);
  d_A[pos] = (d_A[pos] * inv) % prime;
}

__device__ int d_bit_reverse(int x, int n){
  int ans = 0;
  for (int i = 0; i < n; i++)
    if ((x >> i) & 1)
      ans |= ((1 << (n - i - 1)));
  return ans;
}

__global__ void bit_reverse_copy_parallel(LL *a, LL *A, int n, int length){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= length)
      return;
  A[d_bit_reverse(pos, n)] = a[pos];
}


void fft_parallel (LL *d_a, LL *d_A, int dir, LL prime, LL ln, int length) {
  dim3 dim_grid((length + THPB - 1) / THPB, 1, 1);
  dim3 dim_block(THPB, 1, 1);

  bit_reverse_copy_parallel <<< dim_grid, dim_block >>> (d_a, d_A, ln, length);

  for (int s = 1; s <= ln; s++){
    fft_kernel<<<dim_grid, dim_block>>> (d_A, dir, prime, ln, length, s);
  }

  if (dir < 0){
    divide_parallel<<< dim_grid, dim_block >>>(d_A, prime, length);
  }
}

void convolution_gpu(LL *a, LL *b, LL *A, LL prime, LL basew, int length){
  int ln = ceil(log2(float(length)));
  LL *powers = (LL *) malloc (sizeof (LL) * ln);
  compute_powers(powers, ln, basew, prime);

  dim3 dim_grid((length + THPB - 1) / THPB, 1, 1);
  dim3 dim_block(THPB, 1, 1);

  LL *d_a, *d_b, *d_A, *d_B;
  cudaMalloc(&d_a, length * sizeof(LL));
  cudaMalloc(&d_b, length * sizeof(LL));
  cudaMalloc(&d_A, length * sizeof(LL));
  cudaMalloc(&d_B, length * sizeof(LL));

  cudaMemcpy (d_a, a, length * sizeof (LL), cudaMemcpyHostToDevice);
  cudaMemcpy (d_b, b, length * sizeof (LL), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (g_powers, powers, ln * sizeof (LL));

  fft_parallel(d_a, d_A, 1, prime, ln, length);
  fft_parallel(d_b, d_B, 1, prime, ln, length);

  convolution_parallel<<< dim_grid, dim_block >>> (d_A, d_B, prime, length);

  fft_parallel(d_A, d_B, -1, prime, ln, length);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy (A, d_B, length * sizeof (LL), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_A);
  cudaFree(d_B);
  free(powers);
}

void convolution(LL *a, LL *b, LL *A, LL prime, LL basew, int length) {
  long long *B = (long long *) malloc(length * sizeof (long long));
  fft(a, A, 1, prime, basew, length);
  fft(b, B, 1, prime, basew, length);
  for (int i = 0; i < length; ++i){
    A[i] = (A[i] * B[i]) % prime;
  }
  memcpy(B, A, length * sizeof (long long));
  fft(B, A, -1, prime, basew, length);
  free(B);
}

void fft_gpu(LL *a, LL *A, int dir, LL prime, LL basew, int length){
  int ln = ceil(log2(float(length)));

  LL *powers = (LL *) malloc (sizeof (LL) * ln);
  compute_powers(powers, ln, basew, prime);

  LL *d_a, *d_A;
  cudaMalloc(&d_a, length * sizeof(LL));
  cudaMalloc(&d_A, length * sizeof(LL));

  cudaMemcpy (d_a, a, length * sizeof (LL), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (g_powers, powers, ln * sizeof (LL));

  fft_parallel(d_a, d_A, dir, prime, ln, length);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy (A, d_A, length * sizeof (LL), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_A);
  free(powers);
}

bool cmp_vectors (LL *A, LL *B, int length){
  for (int i = 0; i < length; i++){
    if (A[i] != B[i]){
      printf("\nDifference found at %d : %lld !=  %lld\n", i, A[i], B[i]);
      return false;
    }
  }
  return true;
}


int main(int argc, char **argv) {
  if (argc < 5) {
    printf("Usage %s >receiver_endpoint sender_endpoint gpu_id worker_id\n", argv[0]);
    puts("\tTake care with the '>' at the beginin of receiver endpoint.");
    exit(0);
  }

  int id = 0, device_id = 0;

  device_id = atoi(argv[3]);
  id = atoi(argv[4]);

  zsock_t *receiver = zsock_new_pull(argv[1]);
  zsock_t *sender   = zsock_new_push(argv[2]);

  assert(receiver);
  assert(sender);

  int device_count;
  gpuErrchk( cudaGetDeviceCount(&device_count) );

  while (1) {
    // puts("Waiting for messages");
    zmsg_t *message = zmsg_recv(receiver);
    // zmsg_print(message);
    zframe_t *frame = zmsg_next(message);
    LL prime = *((LL *) zframe_data(frame));
    frame = zmsg_next(message);
    LL basew= *((LL *) zframe_data(frame));
    frame = zmsg_next(message);
    int length = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    long long *data = (long long *) zframe_data(frame);
    frame = zmsg_next(message);
    long long *data2 = (long long *) zframe_data(frame);

    // printf("Worker %d solving mod: %lld and length %d\n", id, prime, length);

    if (device_id >= 0 && device_id < device_count){
      gpuErrchk( cudaSetDevice(device_id) );
      // printf("Device selected manually: %d\n", device_id);
    } else {
      printf("Device id not in Range, using default device\n");
    }

    LL *A = (LL*) malloc (sizeof (LL) * length);
    LL *B = (LL*) malloc (sizeof (LL) * length);
    clock_t begin = clock();
    convolution_gpu(data, data2, A, prime, basew, length);
    clock_t end = clock();
    printf("%.10lf\n", double(end - begin) / CLOCKS_PER_SEC);

    begin = clock();
    convolution(data, data2, B, prime, basew, length);
    end = clock();
    printf("%.10lf\n", double(end - begin) / CLOCKS_PER_SEC);

    cmp_vectors(A,B, length);
    // if (cmp_vectors(A, B, length))
      // printf("correct\n");

    zmsg_t *ans = zmsg_new();
    zmsg_addmem(ans, &prime, sizeof (long long));
    zmsg_addmem(ans, &length, sizeof (int));
    zmsg_addmem(ans, A, length * sizeof (long long));
    zmsg_send(&ans, sender);

    zmsg_destroy(&message);
    free(B);
  }

  // Sorry my friend, this code will be unreachable.
  zsock_destroy(&receiver);
  zsock_destroy(&sender);
  return 0;
}
