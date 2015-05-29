#include <czmq.h>
#include <stdio.h>

int main(int arg, char **argv) {
  void *context = zmq_ctx_new();
  puts("entra");
  zmq_ctx_destroy(context);
  return 0;
}

