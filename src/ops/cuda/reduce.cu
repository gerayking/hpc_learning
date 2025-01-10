#include <cuda_runtime.h>
#include <stdio.h>
#include "ops/reduce.h"
/*
reduce_kernel_baseline 是reduce的核函数，输入是input，输出是output，n是输入的维度
input shape: [n]
output shape: [1]
n: 输入的维度
*/
template<int BLOCK_SIZE>
__global__ void reduce_kernel_baseline(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if(tid % (2*s) == 0) {
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v1(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE]; 
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if(threadIdx.x < blockDim.x/(2*s)) {
            int index = threadIdx.x * 2 * s;
            if(index < blockDim.x) {
                sdata[index] += sdata[index + s];
            }
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v2(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE]; 
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(int i=blockDim.x/2; i > 0; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v3(float *input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    for(int i=blockDim.x/2; i > 0; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v3_B(float *input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    for(int i=blockDim.x/2; i > 0; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

__device__ void warp_reduce(float *sdata, int tid){
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v4(float *input, float* output, int n){
    volatile __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    for(int i=blockDim.x/2; i > 32; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads(); 
    }
    if(tid < 32) {
        sdata[tid] += sdata[tid + 32];

        sdata[tid] += sdata[tid + 16];

        sdata[tid] += sdata[tid + 8];

        sdata[tid] += sdata[tid + 4]; 

        sdata[tid] += sdata[tid + 2];

        sdata[tid] += sdata[tid + 1];

        // printf("Block %d, Thread %d, Initial sdata[%d] = %f\n", 
        // blockIdx.x, tid, tid, sdata[tid]);
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v5(float *input, float* output, int n){
    
    volatile __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    
    if(tid < 128){ 
        sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if(tid < 64) {
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    }
    if(tid < 32){

        sdata[tid] += sdata[tid + 32];
        
        sdata[tid] += sdata[tid + 16];

        sdata[tid] += sdata[tid + 8];

        sdata[tid] += sdata[tid + 4]; 

        sdata[tid] += sdata[tid + 2];

        sdata[tid] += sdata[tid + 1];

    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

template<int BLOCK_SIZE>
__global__ void reduce_kernel_v6(float *input, float* output, int n){
    const int WARP_SIZE = 32;
    float sum = 0.f;
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sum = input_begin[tid] + input_begin[tid + blockDim.x];
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    __shared__ float warpLevelSums[32];
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;
    if(laneId==0)warpLevelSums[warpId] = sum;
    __syncthreads();
    if(tid < 32){
        sum = (tid < blockDim.x /32) ? warpLevelSums[tid] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    if(tid == 0) output[blockIdx.x] = sum;
}

template __global__ void reduce_kernel_baseline<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v1<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v2<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v3<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v3_B<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v4<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v5<256>(float* input, float* output, int n);
template __global__ void reduce_kernel_v6<256>(float* input, float* output, int n); 

template __global__ void reduce_kernel_baseline<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v1<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v2<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v3<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v3_B<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v4<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v5<128>(float* input, float* output, int n);
template __global__ void reduce_kernel_v6<128>(float* input, float* output, int n); 