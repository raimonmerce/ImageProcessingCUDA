#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock
#include <opencv2/core/cuda/common.hpp>

using namespace std;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst, cv::cuda::GpuMat& tmp, float *d_vec, int ksize);
//void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst, float *d_vec, int ksize);

void getMatrix(float *vec, int ksize, float sigma) {
  float r;
  float s = 2.0 * sigma * sigma;
  int val = (ksize - 1) / 2;
  float total = 0.0;
  float *matrix = new float[ksize*ksize];
  int count = 0;
  for (int x = -val; x <= val; x++) {
    float totalLoop = 0.0;
    for (int y = -val; y <= val; y++) {
      r = sqrt(x * x + y * y);
      float res = (exp(-(r * r) / s)) / (3.14159265359 * s);
      total += res;
      totalLoop += res;
    }
    vec[count] = totalLoop;
    ++count;
  }
  for (int i = 0; i < ksize; ++i)
    vec[i] /= total;
}

int main( int argc, char** argv )
{
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Resized Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::cuda::GpuMat d_img,d_result, d_tmp;
  cv::Mat h_result;
  int ksize = atoi(argv[2]);
  float sigma = atof(argv[3]);
  float *vec = new float[ksize];

  getMatrix(vec, ksize, sigma);
  size_t bytes_n = ksize * sizeof(float);

  for (int i = 0; i < ksize ; ++i ) cout << vec[i] << ", ";
  d_img.upload(h_img);
  int width= d_img.cols;
  int height = d_img.size().height;

  float *d_vec;
  cudaMalloc(&d_vec, bytes_n);
  cudaMemcpy(d_vec, vec, bytes_n, cudaMemcpyHostToDevice);
  cv::imshow("Original Image", h_img);
  
  auto begin = chrono::high_resolution_clock::now();
  const int iter = 10000;
  for (int i=0;i<iter;i++)
    {
      cv::cuda::resize(d_img,d_result,cv::Size(width, height), cv::INTER_CUBIC);
      cv::cuda::resize(d_img,d_tmp,cv::Size(width, height), cv::INTER_CUBIC);
      //startCUDA (d_img, d_result, d_tmp, vec, ksize );
      startCUDA (d_img, d_result, d_tmp, d_vec, ksize );
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
