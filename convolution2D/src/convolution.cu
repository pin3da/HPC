#include "lodepng.cpp"
#include <iostream>

using namespace std;

void magic(const char* filename) {
  vector<unsigned char> image; //the raw pixels
  unsigned int width, height;
  unsigned int error = lodepng::decode(image, width, height, filename, LCT_GREY);
  if(error) cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;

  cout << width << " " << height << endl;
  cout << image.size() << endl;

  error = lodepng::encode("cat_grey.png", image, width, height, LCT_GREY);
  if(error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;
}

int main() {
  magic("../images/cat2.png");
  return 0;
}

