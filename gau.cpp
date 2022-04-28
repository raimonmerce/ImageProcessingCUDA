#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock
#include <opencv2/core/cuda/common.hpp>

using namespace std;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst, float *d_matrix, int ksize);

void getMatrix(float *matrix, int ksize, float sigma) {

  float r;
  float s = 2.0 * sigma * sigma;
  int val = (ksize - 1) / 2;

  // sum is for normalization
  float sum = 0.0;

  // generating 5x5 kernel
  int count = 0;
  for (int x = -val; x <= val; x++) {
    for (int y = -val; y <= val; y++) {
      r = sqrt(x * x + y * y);
      matrix[count] = (exp(-(r * r) / s)) / (3.14159265359 * s);
      sum += matrix[count];
      count++;
    }
  }
  for (int i = 0; i < ksize*ksize; ++i)
    matrix[i] /= sum;
}

int main( int argc, char** argv )
{
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Resized Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::cuda::GpuMat d_img,d_result;
  cv::Mat h_result;

  string mode(argv[2]);

  float *matrix;
  int ksize;
  float sigma;
  size_t bytes_n;

  if (mode == "g") {
    if (argc == 5) {
      ksize = atoi(argv[3]);
      sigma = atof(argv[4]);
      matrix = new float[ksize*ksize];
      getMatrix(matrix, ksize, sigma);
      bytes_n = ksize * ksize * sizeof(float);
    }
    else {
      cout << "Add ksize and sigma" << endl;
      return 0;
    }
  } else {
    ksize = 3;
    matrix = new float[9];
    matrix[0] = 0.0;
    matrix[1] = -1.0;
    matrix[2] = 0.0;
    matrix[3] = -1.0;
    matrix[4] = 4.0;
    matrix[5] = -1.0;
    matrix[6] = 0.0;
    matrix[7] = -1.0;
    matrix[8] = 0.0;
    bytes_n = 9 * sizeof(float);
  }

  for (int i = 0; i < 9 ; ++i ) cout << matrix[i] << ", ";

  d_img.upload(h_img);
  int width= d_img.cols;
  int height = d_img.size().height;
  int elemSize = d_img.elemSize();
  cv::imshow("Original Image", h_img);


  float *d_matrix;

  cudaMalloc(&d_matrix, bytes_n);
  cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);

  cv::imshow("Original Image", h_img);
  
  auto begin = chrono::high_resolution_clock::now();
  const int iter = 10000;
  
  for (int i=0;i<iter;i++)
    {
      cv::cuda::resize(d_img,d_result,cv::Size(width, height), cv::INTER_CUBIC);
      startCUDA (d_img, d_result, d_matrix, ksize );
    }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  d_result.download(h_result);
  
  cv::imshow("Resized Image", h_result);

  cout << diff.count() << endl;
  cout << diff.count()/iter << endl;
  cout << iter/diff.count() << endl;
  
  cv::waitKey();
  
  return 0;
}
