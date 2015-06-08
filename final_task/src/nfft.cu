#include <bits/stdc++.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
using namespace std;

typedef long long int LL;
typedef pair<LL, LL> PLL;

/* The following vector of pairs contains pairs (prime, generator) where the prime has an Nth
 * root of unity for N being a power of two. The generator is a number g s.t g^(p-1)=1 (mod p)
 * but is different from 1 for all smaller powers */
PLL ROU[] = {make_pair(1224736769,330732430), make_pair(1711276033,927759239),
            make_pair(167772161,167489322), make_pair(469762049,343261969),
            make_pair(754974721,643797295), make_pair(1107296257,883865065)};

PLL ROU_2[] = {make_pair(1711276033LL, 1223522572LL), make_pair(1790967809LL, 1110378081LL)};

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


//Number theory fft. The size of a must be a power of 2
void ntfft(vector<LL> &a, int dir, const PLL &root_unity) {
  int n = a.size();
  LL prime = root_unity.first;
  LL basew = mod_pow(root_unity.second, (prime-1) / n, prime);
  if (dir < 0) basew = mod_inv(basew, prime);
  for (int m = n; m >= 2; m >>= 1) {
    int mh = m >> 1;
    LL w = 1;
    for (int i = 0; i < mh; i++) {
      for (int j = i; j < n; j += m) {
        int k = j + mh;
        LL x = (a[j] - a[k] + prime) % prime;
        a[j] = (a[j] + a[k]) % prime;
        a[k] = (w * x) % prime;
      }
      w = (w * basew) % prime;
    }
    basew = (basew * basew) % prime;
  }
  int i = 0;
  for (int j = 1; j < n - 1; j++) {
    for (int k = n >> 1; k > (i ^= k); k >>= 1);
    if (j < i) swap(a[i], a[j]);
  }
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


__global__ void fft_kernel (LL *A, int dir, LL prime, int ln, LL *powers, int size){
  int pos = threadIdx.x + blockDim.x * blockIdx.x;
  for (int s = 1; s <= ln; s++){
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
    __syncthreads();
  }

  if (dir < 0) {
    LL in = d_mod_inv(size, prime);
    for (int i = 0; i < size; i++)
      A[i] = (A[i] * in) % prime;
  }
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

  fft_kernel<<<dimGrid, dimBlock>>> (d_A, dir, prime, ln, d_powers, size);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

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

int main(){
  LL prime = ROU_2[0].first;
  LL basew = ROU_2[0].second;
  int size = 1024;
  LL *a = (LL*) malloc( sizeof (LL) * size);
  LL *A = (LL*) malloc( sizeof (LL) * size);
  LL *B = (LL*) malloc( sizeof (LL) * size);
  for (int i = 0; i < size; i++){
    a[i] = i;
  }

  fft(a, A, 1, prime, basew, size);
  fft_con(a, B, 1, prime, basew, size);

  if (!cmp_vectors (A, B, size))
    cout << "nay :(" << endl;
  else
    cout << "yay :)" << endl;

  for (int i = 0; i < size; i++)
    cout << B[i] << endl;

  free(a);
  free(A);
  free(B);
  return 0;
}
