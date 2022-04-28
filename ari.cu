#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ uchar3 div(uchar3 a, uchar3 b){
  uchar3 res;
  res.x = ((float(a.x)/255.0)/(float(b.x)/255.0))*255.0;
  res.y = ((float(a.y)/255.0)/(float(b.y)/255.0))*255.0;
  res.z = ((float(a.z)/255.0)/(float(b.z)/255.0))*255.0;
  return res;
}

__device__ uchar3 mult(uchar3 a, uchar3 b){
  uchar3 res;
  res.x = ((float(a.x)/255.0)*(float(b.x)/255.0))*255.0;
  res.y = ((float(a.y)/255.0)*(float(b.y)/255.0))*255.0;
  res.z = ((float(a.z)/255.0)*(float(b.z)/255.0))*255.0;
  return res;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src1, const cv::cuda::PtrStep<uchar3> src2, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int op )
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (dst_x < cols && dst_y < rows)
    {
      uchar3 val1 = src1(dst_y, dst_x);
      uchar3 val2 = src2(dst_y, dst_x);
      if (op == 0){
        dst(dst_y, dst_x).x = val1.x+val2.x;
        dst(dst_y, dst_x).y = val1.y+val2.y;
        dst(dst_y, dst_x).z = val1.z+val2.z;
      } else if (op == 1){
        dst(dst_y, dst_x).x = val1.x-val2.x;
        dst(dst_y, dst_x).y = val1.y-val2.y;
        dst(dst_y, dst_x).z = val1.z-val2.z;
      } else if (op == 2){
        dst(dst_y, dst_x) = mult(val1, val2);
      } else {
        dst(dst_y, dst_x) = div(val1, val2);
      }
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src1, cv::cuda::GpuMat& src2, cv::cuda::GpuMat& dst, int op )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src1, src2, dst, dst.rows, dst.cols, op);

}

