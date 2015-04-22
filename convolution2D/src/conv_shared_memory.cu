#include "lodepng.cpp"
#include <iostream>
#include <string>
#include <sstream>

#define CUDA_CALL(F) if( (F) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK() if( (cudaPeekAtLastError()) != cudaSuccess ) \
{printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__-1); exit(-1);}

using namespace std;

#define THPB 32
#define MASK_SIZE 3

__constant__ char g_filter[MASK_SIZE * MASK_SIZE];

__global__ void conv_kernel(unsigned char *image, unsigned char *ans,
                              int width, int height) {
  __shared__ int s_data[THPB + MASK_SIZE - 1][THPB + MASK_SIZE - 1];

  // Load data to shared memory
  const int radius = MASK_SIZE / 2;
  int dest = threadIdx.y * THPB + threadIdx.x,
      destY = dest / (THPB + MASK_SIZE - 1), destX = dest % (THPB + MASK_SIZE - 1),
      srcY = blockIdx.y * THPB + destY - radius,
      srcX = blockIdx.x * THPB + destX - radius,
      src = srcY * width + srcX;
  if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
    s_data[destY][destX] = image[src];
  else
    s_data[destY][destX] = 0;

  dest = threadIdx.y * THPB + threadIdx.x + THPB * THPB;
  destY = dest / (THPB + MASK_SIZE - 1), destX = dest % (THPB + MASK_SIZE - 1);
  srcY = blockIdx.y * THPB + destY - radius;
  srcX = blockIdx.x * THPB + destX - radius;
  src = srcY * width + srcX;
  if (destY < THPB + MASK_SIZE - 1) {
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
      s_data[destY][destX] = image[src];
    else
      s_data[destY][destX] = 0;
  }
  __syncthreads();



  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  if (x > height || y > width)
    return;

  int cur = 0, nx, ny;
  for (int i = 0; i < MASK_SIZE; ++i) {
    for (int j = 0; j < MASK_SIZE; ++j) {
      nx = threadIdx.y + i;
      ny = threadIdx.x + j;
      if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
        cur += s_data[nx][ny] * g_filter[i * MASK_SIZE + j];
      }
    }
  }
  ans[x * width + y] = min(255, max(0, cur));

  __syncthreads();
}

double sequential(unsigned char *image, unsigned char *ans,
    int width, int height, char *filter, int f_size) {
  int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

  f_size = f_size * f_size;
  clock_t start = clock();
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int cur = 0;
      for (int k = 0; k < f_size; ++k) {
        int x = i + dx[k];
        int y = j + dy[k];
        if (x >= 0 && x < height && y >= 0 && y < width) {
          cur += filter[k] * image[x * width + y];
        }
      }
      ans[i * width + j] = min(255, max(0, cur));
    }
  }
  return (clock() - start) / (double) CLOCKS_PER_SEC;
}


double global_memory(unsigned char *image, unsigned char *ans,
    int width, int height, char *filter, int f_size) {
  clock_t start = clock();
  unsigned char *d_image, *d_ans;
  CUDA_CALL(cudaMalloc(&d_image, width * height * sizeof(unsigned char)));
  CUDA_CALL(cudaMalloc(&d_ans, width * height * sizeof(unsigned char)));

  CUDA_CALL(cudaMemcpy(d_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(g_filter, filter, f_size * f_size * sizeof(char)));

  dim3 dim_grid((width + THPB - 1) / THPB, (height + THPB - 1) / THPB, 1);
  dim3 dim_block(THPB, THPB, 1);

  conv_kernel<<< dim_grid, dim_block >>> (d_image, d_ans, width, height);
  CUDA_CHECK();

  CUDA_CALL(cudaMemcpy(ans, d_ans, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_image));
  CUDA_CALL(cudaFree(d_ans));
  return (clock() - start) / (double) CLOCKS_PER_SEC;
}

double tiled(unsigned char *image, int width, int height) {
  clock_t start = clock();

  return (clock() - start) / (double) CLOCKS_PER_SEC;
}

double const_memory(unsigned char *image, int width, int height) {
  clock_t start = clock();

  return (clock() - start) / (double) CLOCKS_PER_SEC;
}


void magic(const char* filename) {


  string target(filename);
  stringstream ss(target);
  while(getline(ss, target, '/'));

  vector<unsigned char> image; //the raw pixels
  unsigned int width, height;
  unsigned int error = lodepng::decode(image, width, height, filename, LCT_GREY);
  if(error) cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;

  unsigned char *data = (unsigned char *) malloc(width * height * sizeof (unsigned char));
  unsigned char *ans  = (unsigned char *) malloc(width * height * sizeof (unsigned char));

  for (int i = 0; i < image.size(); ++i)
    data[i] = image[i];

  char sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  // char sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  // double sec_time = sequential(data, ans, width, height, sobel_x, 3);
  double sec_time = sequential(data, ans, width, height, sobel_y, 3);
  image = vector<unsigned char>(ans, ans +  width * height);
  error = lodepng::encode(string("seq_") + target, image, width, height, LCT_GREY);
  if(error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;

  double glm_time = global_memory(data, ans, width, height, sobel_y, 3);
  image = vector<unsigned char>(ans, ans +  width * height);
  error = lodepng::encode(string("par_") + target, image, width, height, LCT_GREY);
  if(error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;

  cout << image.size() << '\t' << sec_time << '\t' << glm_time  << endl;
  free(data);
  free(ans);
}

int main(int argc, char **argv) {
  if (argc > 1)
    magic(argv[1]);
  else
    magic("../images/cat1.png");
  return 0;
}

