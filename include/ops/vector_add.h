#pragma once
#include <cuda_runtime.h>

// 主机端调用函数声明
cudaError_t vectorAdd(const float* h_a, const float* h_b, float* h_c, int n);

// 内核函数声明
__global__ void vector_add_kernel_v1(float* a, float* b, float* c, int n);

__global__ void vector_add_kernel_v0(float* a, float* b, float* c, int n);

__global__ void vector_add_kernel_v2(float* a, float* b, float* c, int n);

__global__ void vector_add_kernel_v3(float* a, float* b, float* c, int n, int stride);