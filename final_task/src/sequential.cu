#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <map>

using namespace std;

typedef long long int LL;

pair<long long, long long> ROU[] = {make_pair(1224736769,330732430), make_pair(1711276033,927759239),
            make_pair(167772161,167489322), make_pair(469762049,343261969),
            make_pair(754974721,643797295), make_pair(1107296257,883865065)};

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

void crt(LL **data, LL *mod, int num_taks, int length, LL *ans) {
  LL n = 1;
  for (int i = 0; i < num_taks; ++i)
    n = n * mod[i];

  for (int i = 0; i < length; ++i) {
    LL z = 0;
    for (int j = 0; j < num_taks; ++j) {
      LL tmp = (data[j][i] * (n / mod[j])) % n;
      tmp = (tmp * mod_inv(n / mod[j], mod[j])) % n;
      z = (z + tmp) % n;
    }
    ans[i] = z;
  }
}


int main(int argc, char **argv){
  if (argc < 2){
    cout << "Usage ./sequential num_tasks" << endl;
    return 0;
  }

  const int length = 1024 * 1024;
  const int num_tasks = atoi(argv[1]);

  LL *data = (LL *) malloc (length * sizeof(LL));
  LL *data2 = (LL *) malloc(length * sizeof(LL));
  for (int i = 0; i < length; ++i){
    data[i] = i;
    data2[i] = length - i;
  }

  LL **full_data = (LL **) malloc (num_tasks * sizeof (LL *));
  LL *mods = (LL *) malloc (num_tasks * sizeof(LL));
  clock_t begin = clock();
  for (int i = 0; i < num_tasks; ++i){
    full_data[i] = (LL *) malloc (length * sizeof(LL));
    convolution (data, data2, full_data[i], ROU[i].first, ROU[i].second, length);
    mods[i] = ROU[i].first;
  }

  LL *ans = (LL *) malloc (length * sizeof(LL));
  crt (full_data, mods, num_tasks, length, ans);
  clock_t end = clock();

  cout << double(end - begin) / CLOCKS_PER_SEC << endl;

  free (data);
  free (data2);
  free (full_data);
  free (mods);
  return 0;
}
