#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <ops/vector_add.h>

#define FLOAT4(x) (*reinterpret_cast<float4*>(&(x)))

__global__ void vector_add_kernel_v0(float* a, float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


// CUDA kernel函数 - 在GPU上执行向量加法
__global__ void vector_add_kernel_v1(float* a, float* b, float* c, int n) {
    
    // 计算当前线程处理的元素索引
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    // 确保线程索引不超过数组边界
    if(idx < n){
        float4 tempA = FLOAT4(a[idx]);
        float4 tempB = FLOAT4(b[idx]);
        float4 tempC;
        tempC.x = tempA.x + tempB.x;
        tempC.y = tempA.y + tempB.y;
        tempC.z = tempA.z + tempB.z;
        tempC.w = tempA.w + tempB.w;
        FLOAT4(c[idx]) = tempC;
    }
}

// CUDA kernel函数 - 在GPU上执行向量加法
__global__ void vector_add_kernel_v2(float* a, float* b, float* c, int n) {
    
    // 计算当前线程处理的元素索引
    int idx = 8 * blockDim.x * blockIdx.x + threadIdx.x * 4;
    // 确保线程索引不超过数组边界
    #pragma unroll
    for(int i=0;i<2;i++){
        if(idx+i*blockDim.x*4 < n){
            float4 tempA = FLOAT4(a[idx+i*blockDim.x*4]);
            float4 tempB = FLOAT4(b[idx+i*blockDim.x*4]);
            float4 tempC;
            tempC.x = tempA.x + tempB.x;
            tempC.y = tempA.y + tempB.y;
            tempC.z = tempA.z + tempB.z;
            tempC.w = tempA.w + tempB.w;
            FLOAT4(c[idx+i*blockDim.x*4]) = tempC;
        }
    }
}
// CUDA kernel函数 - 在GPU上执行向量加法
__global__ void vector_add_kernel_v3(float* a, float* b, float* c, int n, int stride) {
    
    // 计算当前线程处理的元素索引
    int idx = 4 *stride * blockDim.x * blockIdx.x + threadIdx.x * 4;
    // 确保线程索引不超过数组边界
    #pragma unroll
    for(int i=0;i<stride;i++){
        if(idx+i*blockDim.x*4 < n){
            float4 tempA = FLOAT4(a[idx+i*blockDim.x*4]);
            float4 tempB = FLOAT4(b[idx+i*blockDim.x*4]);
            float4 tempC;
            tempC.x = tempA.x + tempB.x;
            tempC.y = tempA.y + tempB.y;
            tempC.z = tempA.z + tempB.z;
            tempC.w = tempA.w + tempB.w;
            FLOAT4(c[idx+i*blockDim.x*4]) = tempC;
        }
    }
}

// 主机端调用函数
cudaError_t vectorAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    if (n == 0) return cudaSuccess;
    float *d_a, *d_b, *d_c;
    cudaError_t err = cudaSuccess;

    // 在GPU上分配内存
    err = cudaMalloc((void**)&d_a, n * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc((void**)&d_b, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return err;
    }
    
    err = cudaMalloc((void**)&d_c, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return err;
    }

    // 将输入数据从主机复制到设备
    err = cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // 配置kernel启动参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    // 启动kernel
    vector_add_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // 检查kernel执行错误
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // 将结果从设备复制回主机
    err = cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    return err;
}
