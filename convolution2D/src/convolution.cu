#include "lodepng.cpp"
#include <iostream>

using namespace std;

double sequential(unsigned char *image, unsigned char *ans,
                       int width, int height, char *filer, int f_size) {
  int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

  time_t start = clock();
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int cur = 0;
      for (int k = 0; k < f_size; ++k) {
        int x = i + dx[k];
        int y = j + dy[k];
        if (x >= 0 && x < height && y >= 0 && y < width) {
          cur += filer[k] * image[x * width + y];
        }
      }
      ans[i * width + j] = min(255, max(0, cur));
    }
  }
  return clock() - start;
}

double global_memory(unsigned char *image, int width, int height, char *filer, int f_size) {
  return 0;
}

double tiled(unsigned char *image, int width, int height) {
  return 0;
}

double const_memory(unsigned char *image, int width, int height) {
  return 0;
}


void magic(const char* filename) {
  vector<unsigned char> image; //the raw pixels
  unsigned int width, height;
  unsigned int error = lodepng::decode(image, width, height, filename, LCT_GREY);
  if(error) cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;

  unsigned char data[width * height];
  unsigned char ans[width * height];
  for (int i = 0; i < image.size(); ++i)
    data[i] = image[i];

  char sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  char sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  // cout << width << " " << height << endl;
  // cout << image.size() << endl;
  // double sec_time = sequential(data, ans, width, height, sobel_x, 3 * 3);
  double sec_time = sequential(data, ans, width, height, sobel_y, 3 * 3);
  // global_memory(data, width, height, sobel_y, 3 * 3);

  image = vector<unsigned char>(ans, ans +  width * height);
  vector<unsigned char> to_file(ans, ans + width * height);

  error = lodepng::encode("cat_grey.png", image, width, height, LCT_GREY);
  if(error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;


}

int main() {
  magic("../images/cat2.png");
  return 0;
}

