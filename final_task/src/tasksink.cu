#include <czmq.h>
#include <stdlib.h>
#include <bits/stdc++.h>

using namespace std;

const int THPB = 1024;

__device__ void ext_euclid(long long a, long long b, long long &x, long long &y, long long &g) {
  x = 0, y = 1, g = b;
  long long m, n, q, r;
  for (long long u = 1, v = 0; a != 0; g = a, a = r) {
    q = g / a, r = g % a;
    m = x - u * q, n = y - v * q;
    x = u, y = v, u = m, v = n;
  }
}

__device__ long long mod_inv(long long n, long long m) {
  long long x, y, gcd;
  ext_euclid(n, m, x, y, gcd);
  if (gcd != 1)
    return 0;
  if (x < 0)
    return x += m;
  else if (x >= m)
    return x % m;
  return x;
}


__global__ void parallel_crt(long long *data, long long *mod, int num_tasks, int length, long long n, long long *ans) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= length)
    return;

  long long z = 0;
  for (int j = 0; j < num_tasks; ++j) {
    long long tmp = (data[(j * length) + i] * (n / mod[j])) % n;
    tmp = (tmp * mod_inv(n / mod[j], mod[j])) % n;
    z = (z + tmp) % n;
  }
  if (z < 0)
    z += n;
  ans[i] = z;
}

void crt(long long **data, long long *mod, int num_tasks, int length, long long *ans) {
  long long *d_data, *d_mod, *d_ans;
  cudaMalloc ((void **) &d_data, length * num_tasks * sizeof (long long));
  cudaMalloc ((void **) &d_mod, num_tasks * sizeof (long long));
  cudaMalloc ((void **) &d_ans, length * sizeof (long long));

  for (int i = 0; i < num_tasks; ++i)
    cudaMemcpy (d_data + length * i, data[i], length * sizeof (long long), cudaMemcpyHostToDevice);

  cudaMemcpy (d_mod, mod, num_tasks * sizeof (long long), cudaMemcpyHostToDevice);

  int num_blocks = (length + THPB - 1) / THPB;

  dim3 dim_block(THPB, 1, 1);
  dim3 dim_grid(num_blocks, 1, 1);


  long long n = 1;
  for (int i = 0; i < num_tasks; ++i)
    n = n * mod[i];

  parallel_crt <<< dim_grid, dim_block >>> (d_data, d_mod, num_tasks, length, n, d_ans);
  cudaDeviceSynchronize();

  cudaMemcpy (ans, d_ans, length * sizeof (long long), cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_mod);
  cudaFree(d_ans);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage %s receiver_endpoint\n", argv[0]);
    exit(0);
  }

  zsock_t *receiver = zsock_new_pull(argv[1]);

  // Wait for start of batch
  // char *message = zstr_recv(receiver);
  // puts(message);
  // zstr_free(&message);

  int num_tasks = 6;

  long long *mod   = (long long *) malloc (num_tasks * sizeof (long long *));
  long long **data = (long long **) malloc (num_tasks * sizeof (long long *));

  int length = 0;
  for (int i = 0; i < num_tasks; ++i) {
    zmsg_t *message = zmsg_recv(receiver);
    // zmsg_print(message);
    zframe_t *frame = zmsg_next(message);
    mod[i] = *((long long *) zframe_data(frame));
    frame = zmsg_next(message);
    length = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    data[i] = (long long *) malloc (length * sizeof (long long));

    long long *data_ptr = (long long *) zframe_data(frame);

    memcpy(data[i], data_ptr, length * sizeof (long long));
    /*printf("Using mod: %lld, len %d\n", mod[i], length);
     for (int j = length - 1; length - j < 20; --j)
      printf("%lld ", data[i][j]);
    puts("");
    */
    zmsg_destroy(&message);
  }

  long long *ans = (long long *) malloc ( length * sizeof (long long));

  clock_t start = clock();
  crt(data, mod, num_tasks, length, ans);
  printf("%.10lf\n", (double) (clock() - start) / CLOCKS_PER_SEC);

  /*for (int j = length - 1; length - j < 20; --j)
    printf("%lld ", ans[j]);
  puts("");

  puts("All tasks done"); */

  for (int i = 0; i < num_tasks; ++i)
    free (data[i]);
  free (data);
  free (mod);
  zsock_destroy(&receiver);
  return 0;
}
