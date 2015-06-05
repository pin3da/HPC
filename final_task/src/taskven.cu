#include <czmq.h>
#include <stdlib.h>
#include <string.h>
#include <map>

using namespace std;

pair<long long, long long> ROU[] = {make_pair(1224736769,330732430), make_pair(1711276033,927759239),
            make_pair(167772161,167489322), make_pair(469762049,343261969),
            make_pair(754974721,643797295), make_pair(1107296257,883865065)};

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

  int length = 1024 * 1024, num_tasks = 3;
  long long *data = (long long *) malloc(length * sizeof (long long));
  long long *data2 = (long long *) malloc(length * sizeof (long long));

  for (int i = 0; i < length; ++i) { // Adding some dummy data, replace with meaningful information.
    data[i] = i;
    data2[i] = length - i;
  }

  printf("Press Enter when the workers are ready: ");
  getchar();
  printf("Sending task to workers...\n");

  for (int i = 0; i < num_tasks; ++i) {
    puts("Sending message");
    zmsg_t *message = zmsg_new();
    zmsg_addmem(message, &(ROU[i].first), sizeof (long long));
    zmsg_addmem(message, &(ROU[i].second), sizeof (long long));
    zmsg_addmem(message, &length, sizeof (int));
    zmsg_addmem(message, data, length * sizeof (long long));
    zmsg_addmem(message, data2, length * sizeof (long long));
    int ans = zmsg_send(&message, sender);
    printf("ans : %d\n", ans);
    sleep(1);
    // puts("sent");
  }

  free (data);
  zsock_destroy(&sender);
  zsock_destroy(&sink);
  return 0;
}

