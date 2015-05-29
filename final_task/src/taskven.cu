#include <czmq.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage %s endpoint\n", argv[0]);
    exit(0);
  }

  puts(argv[1]);
  zsock_t *push = zsock_new_push(argv[1]);

  zsock_destroy(&push);
  return 0;
}

