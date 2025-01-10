#ifndef REDUCE_H
#define REDUCE_H

#include <cuda_runtime.h>

// 基础版本的reduce kernel
template<int BLOCK_SIZE>
__global__ void reduce_kernel_baseline(float* input, float* output, int n);

// 优化版本1：改进的线程索引计算
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v1(float* input, float* output, int n);

// 优化版本2：改进的循环结构
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v2(float* input, float* output, int n);

// 优化版本3：每个线程处理两个元素
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v3(float* input, float* output, int n);

// 优化版本3B：与v3相同的实现
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v3_B(float* input, float* output, int n);

// warp级别的reduce辅助函数
__device__ void warp_reduce(float* sdata, int tid);

// 优化版本4：使用warp-level优化
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v4(float* input, float* output, int n);

// 优化版本5：展开循环
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v5(float* input, float* output, int n);

// 优化版本6：使用shuffle指令
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v6(float* input, float* output, int n);


#endif // REDUCE_H