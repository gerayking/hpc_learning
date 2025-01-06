#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <ops/vector_add.h>
// CUDA kernel函数 - 在GPU上执行向量加法
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    

    // 计算当前线程处理的元素索引
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int float4_idx = idx * 4;
    
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    // 确保线程索引不超过数组边界
    if (float4_idx +3 < n) {
        float4 tempA = a4[idx];
        float4 tempB = b4[idx];
        float4 tempC;
        tempC.x = tempA.x + tempB.x;
        tempC.y = tempA.y + tempB.y;
        tempC.z = tempA.z + tempB.z;
        tempC.w = tempA.w + tempB.w;
        c4[idx] = tempC;
    }

    if(float4_idx < n ){
        for(int i = 0; i < 4 && float4_idx + i < n; i++){
            c[float4_idx + i] = a[float4_idx + i] + b[float4_idx + i];
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
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // 检查kernel执行错误
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // 将结果从设备复制回主机
    err = cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    return err;
}
