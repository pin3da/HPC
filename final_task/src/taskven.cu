#include <czmq.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage %s sender_endpoint sink_endpoint\n", argv[0]);
    exit(0);
  }

  char sender_ep[80];
  strcpy(sender_ep, "@");
  strcat(sender_ep, argv[1]);
  puts(sender_ep);
  zsock_t *sender = zsock_new_push(sender_ep);
  zsock_t *sink   = zsock_new_push(argv[2]);

  assert(sender);
  assert(sink);
  // printf("Press Enter when the workers are ready: ");
  getchar();
  printf("Sending task to workers...\n");
  zstr_send(sink, "0");

  int num_tasks = 2;
  for (int i = 0; i < num_tasks; ++i) {
    zstr_send(sender, "Manuel was here");
    puts("sent");
  }

  zsock_destroy(&sender);
  zsock_destroy(&sink);
  return 0;
}

