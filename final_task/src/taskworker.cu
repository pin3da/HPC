#include <czmq.h>
#include <stdlib.h>

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
    int mod = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    int length = *((int *) zframe_data(frame));
    frame = zmsg_next(message);
    int *data = (int *) zframe_data(frame);

    printf("Worker %d solving mod: %d and length %d\n", id, mod, length);
    for (int i = 0; i < length; ++i)
      data[i] = data[i] % mod;

    zmsg_t *ans = zmsg_new();
    zmsg_addmem(ans, &mod, sizeof (int));
    zmsg_addmem(ans, &length, sizeof (int));
    zmsg_addmem(ans, data, length * sizeof (int));
    zmsg_send(&ans, sender);

    zmsg_destroy(&message);
  }

  // Sorry my friend, this code will be unreachable.
  zsock_destroy(&receiver);
  zsock_destroy(&sender);
  return 0;
}

