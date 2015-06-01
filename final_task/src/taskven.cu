#include <czmq.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage %s @sender_endpoint sink_endpoint\n", argv[0]);
    puts("\tTake care with the '@' at the begining of sender_endpoint.");
    exit(0);
  }

  zsock_t *sender = zsock_new_push(argv[1]);
  zsock_t *sink   = zsock_new_push(argv[2]);

  assert(sender);
  assert(sink);

  // zstr_send(sink, "start");

  int length = 10000000, num_tasks = 3;
  int *data = (int *) malloc(length * sizeof (int));
  int *mod  = (int *) malloc(num_tasks * sizeof (int));

  mod[0] = 7901;
  mod[1] = 7907;
  mod[2] = 7919;

  for (int i = 0; i < length; ++i) // Adding some dummy data, replace with a meanful information.
    data[i] = i;

  printf("Press Enter when the workers are ready: ");
  getchar();
  printf("Sending task to workers...\n");

  for (int i = 0; i < num_tasks; ++i) {
    puts("Sending message");
    zmsg_t *message = zmsg_new();
    zmsg_addmem(message, &(mod[i]), sizeof (int));
    zmsg_addmem(message, &length, sizeof (int));
    zmsg_addmem(message, data, length * sizeof (int));
    int ans = zmsg_send(&message, sender);
    printf("ans : %d\n", ans);
    sleep(1);
    // puts("sent");
  }

  free (data);
  free (mod);
  zsock_destroy(&sender);
  zsock_destroy(&sink);
  return 0;
}

