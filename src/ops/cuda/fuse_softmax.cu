#include <cuda_runtime.h>


/*
softmax_kernel 是softmax的核函数，输入是input，输出是output，n是输入的维度
input shape: [batch_size, seq_len, hidden_size]
output shape: [batch_size, seq_len, hidden_size]
n: 输入的维度
*/
template<int BLOCK_SIZE>
__global__ void softmax_kernel(
    float* output_ptr,
    const float* input_ptr,
    const int input_row_stride,
    const int output_row_stride,
    const int n_rows,
    const int n_cols
) {
    // 获取当前行索引
    int row_idx = blockIdx.x;
    if (row_idx >= n_rows) return;

    // 计算当前行的起始位置
    const float* row_start_ptr = input_ptr + row_idx * input_row_stride;
    float* output_row_start_ptr = output_ptr + row_idx * output_row_stride;
    
    // 使用共享内存存储一行数据
    __shared__ float row_data[BLOCK_SIZE];
    __shared__ float row_exp[BLOCK_SIZE];
    
    // 加载数据到共享内存
    for (int tid = threadIdx.x; tid < n_cols; tid += blockDim.x) {
        row_data[tid] = row_start_ptr[tid];
    }
    __syncthreads();
    
    // 找到最大值用于数值稳定性
    float max_val = -INFINITY;
    for (int tid = threadIdx.x; tid < n_cols; tid += blockDim.x) {
        max_val = max(max_val, row_data[tid]);
    }
    // 在块内进行规约以获取最大值
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }
    // 广播最大值
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // 计算指数并存储到共享内存
    float sum = 0.0f;
    for (int tid = threadIdx.x; tid < n_cols; tid += blockDim.x) {
        float exp_val = expf(row_data[tid] - max_val);
        row_exp[tid] = exp_val;
        sum += exp_val;
    }
    
    // 在块内进行规约以获取总和
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    // 广播总和
    sum = __shfl_sync(0xffffffff, sum, 0);
    
    // 计算最终的softmax值并写回全局内存
    for (int tid = threadIdx.x; tid < n_cols; tid += blockDim.x) {
        output_row_start_ptr[tid] = row_exp[tid] / sum;
    }
}
