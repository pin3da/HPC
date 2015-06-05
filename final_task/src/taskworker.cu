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

PLL ROU[] = {make_pair(1711276033LL, 1223522572LL), make_pair(1790967809LL, 1110378081LL)};

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

PLL ext_euclid(LL a, LL b) {
  if (b == 0)
    return make_pair(1,0);
  pair<LL,LL> rc = ext_euclid(b, a % b);
  return make_pair(rc.second, rc.first - (a / b) * rc.second);
}

//returns -1 if there is no unique modular inverse
LL mod_inv(LL x, LL modulo) {
  PLL p = ext_euclid(x, modulo);
  if ( (p.first * x + p.second * modulo) != 1 )
    return -1;
  return (p.first+modulo) % modulo;
}

// Computes ( a ^ exp ) % mod.
LL mod_pow(LL a, LL exp, LL mod) {
  LL ans = 1, base = a;
  while (exp > 0) {
    if (exp & 1)
      ans = (ans * base) % mod;
    base = (base * base) % mod;
    exp >>= 1;
  }
  return ans;
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
    if (dir == -1)
      wm =  mod_inv(wm, prime);

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

  if (dir < 0) {
    LL in = mod_inv(length, prime);
    for (int i = 0; i < length; i++)
      A[i] = (A[i] * in) % prime;
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


__global__ void fft_kernel (LL *A, int dir, LL prime, int ln, LL *powers, int length, int s) {
  int pos = threadIdx.x + blockDim.x * blockIdx.x;
  LL m = (1LL << s);
  LL wm = powers[ln - s];
  LL k = pos * m;
  if (dir == -1)
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
  d_A[pos] = (d_A[pos] * d_B[pos]) % prime;
}

__global__ void divide_parallel(LL *d_A, LL prime, int length) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= length)
    return;
  LL inv = d_mod_inv(length, prime);
  d_A[pos] = (d_A[pos] * inv) % prime;
}

void fft_parallel (LL *d_A, int dir, LL prime, LL ln, LL *d_powers, int length) {
  dim3 dim_grid((length + THPB - 1) / THPB, 1, 1);
  dim3 dim_block(THPB, 1, 1);

  for (int s = 1; s <= ln; s++){
    fft_kernel<<<dim_grid, dim_block>>> (d_A, dir, prime, ln, d_powers, length, s);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
}



void convolution(LL *a, LL *b, LL *A, LL prime, LL basew, int length){
  int ln = ceil(log2(float(length)));
  LL *B = (LL *) malloc(length * sizeof (LL));
  // TODO: parallelize the following two calls.
  bit_reverse_copy(a, A, ln, length);
  bit_reverse_copy(b, B, ln, length);
  LL *powers = (LL *) malloc (sizeof (LL) * ln);
  compute_powers(powers, ln, basew, prime);

  LL *d_A, *d_B, *d_powers;
  cudaMalloc(&d_A, length * sizeof(LL));
  cudaMalloc(&d_B, length * sizeof(LL));
  cudaMalloc(&d_powers, ln * sizeof(LL));

  cudaMemcpy (d_A, A, length * sizeof (LL), cudaMemcpyHostToDevice);
  cudaMemcpy (d_B, B, length * sizeof (LL), cudaMemcpyHostToDevice);
  cudaMemcpy (d_powers, powers, ln * sizeof (LL), cudaMemcpyHostToDevice);

  fft_parallel(d_A, 1, prime, ln, d_powers, length);
  fft_parallel(d_B, 1, prime, ln, d_powers, length);
  dim3 dim_grid((length + THPB - 1) / THPB, 1, 1);
  dim3 dim_block(THPB, 1, 1);
  convolution_parallel<<< dim_grid, dim_block >>> (d_A, d_B, prime, length);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  fft_parallel(d_A, -1, prime, ln, d_powers, length);
  divide_parallel<<< dim_grid, dim_block >>>(d_A, prime, length);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy (A, d_A, length * sizeof (LL), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_powers);
  free(powers);
}

bool cmp_vectors (LL *A, LL *B, int length){
  for (int i = 0; i < length; i++){
    if (A[i] != B[i]){
      cout << A[i] << " " << B[i] << " i: " << i << endl;
      return false;
    }
  }
  return true;
}


int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage %s >receiver_endpoint sender_endpoint <id>\n", argv[0]);
    puts("\tTake care with the '>' at the beginin of receiver endpoint.");
    exit(0);
  }

  int id = 0;
  if (argc > 3)
    id = atoi(argv[3]);

  zsock_t *receiver = zsock_new_pull(argv[1]);
  zsock_t *sender   = zsock_new_push(argv[2]);

  assert(receiver);
  assert(sender);

  while (1) {
    puts("Waiting for messages");
    zmsg_t *message = zmsg_recv(receiver);
    zmsg_print(message);
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

    printf("Worker %d solving mod: %lld and length %d\n", id, prime, length);

    LL *A = (LL*) malloc (sizeof (LL) * length);
    LL *B = (LL*) malloc (sizeof (LL) * length);

    clock_t start = clock();
    convolution(data, data2, A, prime, basew, length);
    printf("%.10f\n", ((double) (clock() - start) / CLOCKS_PER_SEC));

    /*start = clock();
    fft(data, B, 1, prime, basew, length);
    printf("%.10f\n", ((double) (clock() - start) / CLOCKS_PER_SEC));

    if (cmp_vectors(A, B, length)) {
      puts("all right");
    } else {
      puts("):");
    }*/

    zmsg_t *ans = zmsg_new();
    zmsg_addmem(ans, &prime, sizeof (long long));
    zmsg_addmem(ans, &length, sizeof (int));
    zmsg_addmem(ans, A, length * sizeof (long long));
    zmsg_send(&ans, sender);

    zmsg_destroy(&message);
  }

  // Sorry my friend, this code will be unreachable.
  zsock_destroy(&receiver);
  zsock_destroy(&sender);
  return 0;
}
