#include <czmq.h>
#include <stdlib.h>

void ext_euclid(long long a, long long b, long long &x, long long &y, long long &g) {
  x = 0, y = 1, g = b;
  long long m, n, q, r;
  for (long long u = 1, v = 0; a != 0; g = a, a = r) {
    q = g / a, r = g % a;
    m = x - u * q, n = y - v * q;
    x = u, y = v, u = m, v = n;
  }
}

long long mod_inv(long long n, long long m) {
  long long x, y, gcd;
  ext_euclid(n, m, x, y, gcd);
  if (gcd != 1)
    return 0;
  return (x + m) % m;
}

void crt(int **data, int *mod, int num_taks, int length, int *ans) {
  long long n = 1;
  for (int i = 0; i < num_taks; ++i)
    n = n * mod[i];

  for (int i = 0; i < length; ++i) {
    long long z = 0;
    for (int j = 0; j < num_taks; ++j) {
      long long tmp = (data[j][i] * (n / mod[j])) % n;
      tmp = (tmp * mod_inv(n / mod[j], mod[j])) % n;
      z = (z + tmp) % n;
    }
    ans[i] = z;
  }
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

  int num_tasks = 3;

  int *mod   = (int *) malloc (num_tasks * sizeof (int *));
  int **data = (int **) malloc (num_tasks * sizeof (int *));

  int length = 0;
  for (int i = 0; i < num_tasks; ++i) {
    zmsg_t *message = zmsg_recv(receiver);
    zmsg_print(message);
    zframe_t *frame = zmsg_next(message);
    mod[i] = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    length = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    data[i] = (int *) malloc (length * sizeof (int));
    int *data_ptr = (int *) zframe_data(frame);
    memcpy(data[i], data_ptr, length * sizeof (int));
    printf("Using mod: %d, len %d\n", mod[i], length);
    for (int j = length - 1; length - j < 20; --j)
      printf("%d ", data[i][j]);
    puts("");
    zmsg_destroy(&message);
  }

  int *ans = (int *) malloc ( length * sizeof (int));

  crt(data, mod, num_tasks, length, ans);

  puts("All tasks done");

  for (int j = length - 1; length - j < 20; --j)
    printf("%d ", ans[j]);
  puts("");

  for (int i = 0; i < num_tasks; ++i)
    free (data[i]);
  free (data);
  free (mod);
  zsock_destroy(&receiver);
  return 0;
}

