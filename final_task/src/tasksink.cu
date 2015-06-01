#include <czmq.h>
#include <stdlib.h>

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
  for (int i = 0; i < num_tasks; ++i) {
    zmsg_t *message = zmsg_recv(receiver);
    zmsg_print(message);
    zframe_t *frame = zmsg_next(message);
    int mod = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    int length = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    int *data = (int *) zframe_data(frame);
    printf("Using mod: %d, len %d\n", mod, length);
    for (int j = length - 1; length - j < 20; --j)
      printf("%d ", data[j]);
    puts("");
    zmsg_destroy(&message);
  }

  puts("All tasks done");

  zsock_destroy(&receiver);
  return 0;
}

