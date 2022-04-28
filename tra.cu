#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ float3 mult(float3 vec, float mult){
  float3 res;
  res.x = vec.x * mult;
  res.y = vec.y * mult;
  res.z = vec.z * mult;
  return res;
}

__device__ float3 sum(float3 a, float3 b){
  float3 res;
  
  res.x = float(a.x) + float(b.x);
  res.y = float(a.y) + float(b.y);
  res.z = float(a.z) + float(b.z);
  return res;
}

__device__ float3 convertFloat(uchar3 a){
  float3 res;
  unsigned char rf = a.x;
  unsigned char gf = a.y;
  unsigned char bf = a.z;
  res.x = rf;
  res.y = gf;
  res.z = bf;
  return res;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float scaleFactor )
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  float X = float(dst_y) / scaleFactor;
  float Y = float(dst_x) / scaleFactor;
  uchar3 res;
  if ((X >= 0.0) && ((X + 1.0) < float(rows)) && (Y >= 0.0) && ((Y + 1.0) < float(cols))) {
    float3 BLVal = convertFloat(src(int(X), int(Y + 1)));
    float3 BRVal = convertFloat(src(int(X + 1), int(Y + 1)));
    float3 TLVal = convertFloat(src(int(X), int(Y)));
    float3 TRVal = convertFloat(src(int(X + 1), int(Y + 1)));

    float3 top = sum(mult(TRVal, X - float(int(X))), mult(TLVal, float(int(X + 1)) - X));
    float3 bot = sum(mult(BRVal, X - float(int(X))), mult(BLVal, float(int(X + 1)) - X));
    float3 resF = sum(mult(top, Y - float(int(Y))), mult(bot, float(int(Y + 1)) - Y));
    unsigned char rf = resF.x;
    unsigned char gf = resF.y;
    unsigned char bf = resF.z;

    dst(dst_y, dst_x).x = rf;
    dst(dst_y, dst_x).y = gf;
    dst(dst_y, dst_x).z = bf;
  }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float scaleFactor )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, scaleFactor);

}

