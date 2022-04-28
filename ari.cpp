#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void startCUDA ( cv::cuda::GpuMat& src1, cv::cuda::GpuMat& src2, cv::cuda::GpuMat& dst, int op );

int main( int argc, char** argv )
{
  cv::namedWindow("Original Image 1", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Original Image 2", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Resized Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img1 = cv::imread(argv[1]);
  cv::Mat h_img2 = cv::imread(argv[2]);
  int op = atoi(argv[3]); //+,-,*,/
  cv::cuda::GpuMat d_img1, d_img2,d_result;
  cv::Mat h_result;


  d_img1.upload(h_img1);
  d_img2.upload(h_img2);
  int width= d_img1.cols;
  int height = d_img1.size().height;

  cv::imshow("Original Image 1", h_img1);
  cv::imshow("Original Image 2", h_img2);
  
  auto begin = chrono::high_resolution_clock::now();
  const int iter = 10000;
  
  for (int i=0;i<iter;i++)
    {
      cv::cuda::resize(d_img1, d_result,cv::Size(width, height), cv::INTER_CUBIC);
      cv::cuda::resize(d_img2, d_result,cv::Size(width, height), cv::INTER_CUBIC);
      startCUDA ( d_img1, d_img2,d_result, op );
      //     cv::imshow("Resized Image", d_result);
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
