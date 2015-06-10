#include <random>
#include <iostream>

using namespace std;

/*
 * Pseudo random array generator for convolution
 * @param length
 * result : two arrays of size 'length' with elements between 1 and 'length'
 * */

int main (int argc, char **argv) {
  if (argc < 2) {
    printf ("Usage : %s length\n", argv[0]);
  }

  int length = atoi(argv[1]);
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dist(1, length);
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < length; ++i) {
      if (i) cout << " ";
     cout << dist(gen);
    }
    cout  << endl;
  }
}
