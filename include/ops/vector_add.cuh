#pragma once
#include <cuda_runtime.h>

// 主机端调用函数声明
cudaError_t vectorAdd(const float* h_a, const float* h_b, float* h_c, int n);