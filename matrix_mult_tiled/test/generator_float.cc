using namespace std;
#include <bits/stdc++.h>

#define TC 100
#define MV 10
#define MM 2048


int main() {
  random_device rd;
  mt19937 gen(rd());

  printf("%d\n", TC);
  int delta = MM / TC;
  int MN = 0;
  for (int i = 0; i < TC; ++i) {
    MN += delta;
    uniform_int_distribution<> dis(min(10, MN), MN);
    uniform_int_distribution<> val(0, MV);

    int m = dis(gen), n = dis(gen), o = dis(gen);

    printf("%d %d %d\n", m, n, o);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j)
        printf("%.10f ", static_cast<float>(val(gen)) / static_cast<float>(MN));
      puts("");
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < o; ++j)
        printf("%.10f ", static_cast<float>(val(gen)) / static_cast<float>(MN));
      puts("");
    }

  }

  return 0;
}
