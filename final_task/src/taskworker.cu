#include <czmq.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

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

void bit_reverse_copy(LL *a, LL *A, int n, int size) {
  for (int i = 0; i < size; i++)
    A[bit_reverse(i, n)] = a[i];
}

void compute_powers(LL *powers, int ln, LL basew, LL prime){
  powers[0] = basew;
  for (int i = 1; i < ln; i++){
    powers[i] = (powers[i - 1] * powers[i - 1]) % prime;
  }
}

void fft(LL *a, LL *A, int dir, LL prime, LL basew, int size) {
  int ln = ceil(log2(float(size)));
  bit_reverse_copy(a, A, ln, size);
  LL *powers = (LL*) malloc (sizeof (LL) * ln);
  compute_powers(powers, ln, basew, prime);

  for (int s = 1; s <= ln; s++) {
    long long m = (1LL << s);
    LL wm = powers[ln - s];
    if (dir == -1)
      wm =  mod_inv(wm, prime);

    for (int k = 0; k < size; k += m) {
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
    LL in = mod_inv(size, prime);
    for (int i = 0; i < size; i++)
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


__global__ void fft_kernel (LL *A, int dir, LL prime, int ln, LL *powers, int size, int s) {
  int pos = threadIdx.x + blockDim.x * blockIdx.x;
  LL m = (1LL << s);
  LL wm = powers[ln -s];
  int k = pos * m;
  if (dir == -1)
    wm = d_mod_inv (wm, prime);
  if (k >= size)
    return;
  else{
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

  /*if (dir < 0) {
    LL in = d_mod_inv(size, prime);
    for (int i = 0; i < size; i++)
    A[i] = (A[i] * in) % prime;
    }*/
}

void fft_con(LL *a, LL *A, int dir, LL prime, LL basew, int size){
  int ln = ceil(log2(float(size)));
  bit_reverse_copy(a, A, ln, size);
  LL *powers = (LL*) malloc (sizeof (LL) * ln);
  compute_powers(powers, ln, basew, prime);

  LL *d_A, *d_powers;
  cudaMalloc(&d_A, size * sizeof(LL));
  cudaMalloc(&d_powers, ln * sizeof(LL));

  cudaMemcpy (d_A, A, size * sizeof (LL), cudaMemcpyHostToDevice);
  cudaMemcpy (d_powers, powers, ln * sizeof (LL), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(float(size / 1024.0)), 1, 1);
  dim3 dimBlock(1024, 1, 1);

  for (int s = 1; s <= ln; s++){
    fft_kernel<<<dimGrid, dimBlock>>> (d_A, dir, prime, ln, d_powers, size, s);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  cudaMemcpy (A, d_A, size * sizeof (LL), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_powers);
  free(powers);

}

bool cmp_vectors (LL *A, LL *B, int size){
  for (int i = 0; i < size; i++){
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

    printf("Worker %d solving mod: %lld and length %d\n", id, prime, length);

    LL *A = (LL*) malloc (sizeof (LL) * length);
    LL *B = (LL*) malloc (sizeof (LL) * length);
    fft_con (data, A, 1, prime, basew, length);
    fft(data, B, 1, prime, basew, length);
    if (cmp_vectors(A, B, length)) {
      puts("all right");
    } else {
      puts("):");
    }

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
