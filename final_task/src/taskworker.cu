#include <czmq.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage %s receiver_endpoint sender_endpoint <id>\n", argv[0]);
    exit(0);
  }

  int id = 0;
  if (argc > 3)
    id = atoi(argv[3]);

  char receiver_ep[80];
  strcpy(receiver_ep, ">");
  strcat(receiver_ep, argv[1]);

  puts(receiver_ep);
  puts(argv[2]);

  zsock_t *receiver = zsock_new_pull(receiver_ep);
  zsock_t *sender   = zsock_new_push(argv[2]);

  // while (1) {
    char *message = zstr_recv(receiver);
    printf("Message on worker %d: %s\n", id, message);
    char buffer[50];
    sprintf(buffer, "Task done on woeker %d", id);
    zstr_send(sender, buffer);
    zstr_free(&message);
  // }

  zsock_destroy(&receiver);
  zsock_destroy(&sender);
  return 0;
}

