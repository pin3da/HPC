#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

struct entry{
  long long size;
  double seq, global, shared, constant;
  entry() {}
  entry(long long a, double b, double c, double d, double e) :
    size(a),
    seq(b),
    global(c),
    shared(d),
    constant(e)
  {}
  bool operator < (const entry &e) const {
    return size < e.size;
  }
};

int main() {

  string line;
  map<string, vector<entry> > times;
  long long size;
  double seq, global, shared, constant;
  while (cin >> line) {
    cin >> size >> seq >> global;
    if (line == "../images/cat9.png") continue;
    cin >> size >> seq >> shared;
    cin >> size >> seq >> constant;
    times[line].push_back(entry(size, seq, global, shared, constant));
  }

  cout << "Size\tSequential Time\tGlobal Memory\tShared Memory\tConstant Memory" << endl;
  vector<entry> ans;
  for (map<string, vector<entry> >::iterator it = times.begin(); it != times.end(); ++it) {
    line = it->first;
    vector<entry> &t = it->second;
    seq = global = shared = constant = 0;
    for (int i = 0; i < t.size(); ++i) {
      size = t[i].size;
      seq += t[i].seq;
      global += t[i].global;
      shared += t[i].shared;
      constant += t[i].constant;
    }
    seq /= (double)t.size();
    global /= (double)t.size();
    shared /= (double)t.size();
    constant /= (double)t.size();
    ans.push_back(entry(size, seq, global, shared, constant));
  }
  sort(ans.begin(), ans.end());
  for (int i = 0; i < ans.size(); ++i) {
    cout << ans[i].size << '\t' << ans[i].seq << '\t' << ans[i].global << '\t' << ans[i].shared << '\t' << ans[i].constant << endl;
  }
  return 0;
}

