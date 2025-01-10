#pragma once
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void softmax_kernel(
    float* output_ptr,
    const float* input_ptr,
    const int input_row_stride,
    const int output_row_stride,
    const int n_rows,
    const int n_cols
);