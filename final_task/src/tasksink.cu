#include <czmq.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage %s receiver_endpoint\n", argv[0]);
    exit(0);
  }

  zsock_t *receiver = zsock_new_pull(argv[1]);

  // Wait for start of batch
  char *message = zstr_recv(receiver);
  zstr_free(&message);

  int num_tasks = 2;
  for (int i = 0; i < num_tasks; ++i) {
    char *message = zstr_recv(receiver);
    puts(message);
    zstr_free(&message);
  }

  puts("All tasks done");

  zsock_destroy(&receiver);
  return 0;
}

